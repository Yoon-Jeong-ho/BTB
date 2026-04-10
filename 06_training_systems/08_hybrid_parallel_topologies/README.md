# 08 Hybrid Parallel Topologies

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조**이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
`06_training_systems/04_fsdp_checkpointing_and_offload`, `05_tensor_parallelism`, `06_pipeline_parallelism`, `07_data_parallel_grad_accumulation`까지 오면 이제 중요한 질문은 각 기법을 따로 설명하는 것이 아니라, **실제 대형 모델 학습에서 이 축들을 어떻게 함께 묶을 것인가**다. 현실의 LLM/멀티모달 모델 학습은 data parallel만으로도, tensor parallel만으로도, pipeline parallel만으로도, FSDP만으로도 끝나지 않는다. 모델 상태 메모리, activation 메모리, intra-layer 연산량, stage bubble, global batch 운영, checkpoint 복구 제약이 동시에 얽히기 때문이다.

그래서 hybrid parallel topology는 단순한 "옵션 조합"이 아니라, **모델 규모와 하드웨어 배치를 연결하는 설계 문제**로 봐야 한다. 어느 축을 node 안에 둘지, 어느 축을 node 사이에 둘지, 통신이 잦은 축을 빠른 interconnect 위에 올릴지, memory-saving 축과 throughput 축을 어떤 비율로 섞을지 결정하는 감각이 필요하다. 이 단위는 바로 그 감각을 만들기 위해, 여러 parallelism 축을 한 장의 topology 그림으로 합쳐 읽는 연습을 한다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- data / tensor / pipeline / FSDP 조합 직관과 설계 기준을 정리한 `THEORY.md`
- 선행 개념 체크리스트 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 실습 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - topology 후보별 mesh 구성표와 rank 역할 요약
  - memory / throughput / communication budget 비교표
  - 모델 규모 대비 하드웨어 매핑 체크리스트
  - 병목 지점 및 failure signature 관찰 메모

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. 먼저 한 가지 모델 규모(예: 30B, 70B, 100B+)와 한 가지 클러스터 제약(예: 8 GPU 단일 노드, 64 GPU 다중 노드)을 정하고, 어떤 병목이 먼저 오는지부터 분류한다.
2. data parallel, tensor parallel, pipeline parallel, FSDP가 각각 해결하려는 문제가 무엇인지 다시 나눈 뒤, **어떤 축은 상태 메모리를 줄이고 어떤 축은 active compute를 나누며 어떤 축은 시간축 실행을 다시 짠다**는 점을 한 장의 표로 정리한다.
3. device mesh를 가정해 tensor parallel은 node 내부의 빠른 링크 위에, pipeline parallel은 stage 경계 기준으로, data parallel/FSDP는 replica·shard 그룹 기준으로 어떻게 배치할지 후보 topology를 그려 본다.
4. 각 후보에 대해 all-reduce, all-gather, reduce-scatter, activation send/recv가 어디서 가장 많이 생길지 추정하고, 통신이 빠른 링크와 느린 링크를 어떻게 타게 되는지 비교한다.
5. global batch, microbatch, gradient accumulation, checkpoint save/load, optimizer state 배치까지 함께 넣어 보며 **단순히 돌아가는지**가 아니라 **운영 가능한지**를 판단한다.
6. 마지막에는 어떤 topology가 특정 모델/클러스터에서 더 현실적인지 선택 이유를 적고, 다음 단위 `06_training_systems/09_profiling_monitoring_and_failure_recovery`에서 실제 병목과 장애를 어디서 관찰할지 연결한다.

## 이 단위에서 특히 볼 질문
- hybrid parallel topology는 왜 "parallelism을 다 켠다"가 아니라, 모델 병목과 하드웨어 제약을 맞추는 설계 문제로 봐야 하는가?
- data parallel, tensor parallel, pipeline parallel, FSDP는 각각 무엇을 나누며, 서로 겹치는 부분과 독립 축은 어디인가?
- 통신이 많은 축은 왜 node 내부 NVLink/NVSwitch 같은 빠른 링크 쪽에 두고, 상대적으로 덜 민감한 축은 node 간으로 밀어내는 경우가 많은가?
- 모델 크기·sequence length·global batch 목표가 달라지면 topology 선택 기준은 어떻게 달라지는가?
- memory fit, throughput, implementation complexity, checkpoint portability는 왜 동시에 최적화되지 않는가?
- 실제 운영에서 topology가 잘못 설계되었을 때는 어떤 로그/프로파일/메모리 흔적으로 먼저 드러나는가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 06_training_systems/08_hybrid_parallel_topologies/scratch_lab.py
{
  "status": "sample",
  "model": {
    "name": "decoder_only_llm",
    "params_b": 70,
    "sequence_length": 8192,
    "target_global_batch": 1024
  },
  "hardware": {
    "nodes": 8,
    "gpus_per_node": 8,
    "intra_node_link": "NVLink",
    "inter_node_link": "InfiniBand"
  },
  "candidate_topology": {
    "data_parallel": 8,
    "tensor_parallel": 4,
    "pipeline_parallel": 2,
    "fsdp_mode": "hybrid_shard",
    "microbatch_per_pipeline": 4,
    "grad_accum_steps": 4
  },
  "fit_summary": {
    "per_rank_param_state_gb": 11.8,
    "activation_peak_gb": 18.4,
    "estimated_tokens_per_step": 8388608,
    "primary_risk": "tensor-parallel collectives across node boundary if mesh is misaligned"
  },
  "notes": [
    "sample numbers for intuition only",
    "expected output/sample shape only"
  ]
}

$ python 06_training_systems/08_hybrid_parallel_topologies/framework_lab.py
{
  "status": "sample",
  "topology_candidates": [
    {
      "name": "tp4_pp2_dp8_fsdp_hybrid",
      "best_for": "fast intra-node links, moderate pipeline depth",
      "communication_hotspots": ["tp all-reduce", "fsdp all-gather"],
      "memory_notes": ["optimizer state remains sharded", "activation pressure depends on microbatch"]
    },
    {
      "name": "tp2_pp4_dp8_fsdp_full_shard",
      "best_for": "deeper model partition with smaller per-stage memory",
      "communication_hotspots": ["pipeline send/recv", "checkpoint reshaping"],
      "memory_notes": ["smaller stage footprint", "higher bubble/load-balance sensitivity"]
    }
  ],
  "selection_summary": {
    "preferred_candidate": "tp4_pp2_dp8_fsdp_hybrid",
    "reason": [
      "keeps tensor-parallel traffic inside node",
      "limits pipeline bubble compared with deeper stage split",
      "preserves sharded optimizer-state memory savings"
    ],
    "profiling_focus": [
      "collective overlap",
      "stage imbalance",
      "checkpoint save/load portability"
    ]
  }
}
```

핵심은 특정 숫자를 외우는 것이 아니라, **모델 규모를 어떤 병렬화 축 조합으로 하드웨어에 끼워 넣는지**, **그때 통신과 메모리 부담이 어느 경계로 이동하는지**, **운영 가능한 checkpoint / batch / schedule 계약이 무엇인지**를 읽는 것이다.

## 다음 단위와의 연결
다음 단위 `06_training_systems/09_profiling_monitoring_and_failure_recovery`에서는 여기서 설계한 hybrid topology가 실제로 돌 때, 어느 링크가 막히는지, 어느 stage가 놀고 있는지, 어떤 rank에서 OOM이나 timeout이 먼저 나는지, checkpoint 복구가 왜 꼬이는지 같은 **운영 흔적**을 본격적으로 다룬다.

즉 이 단위가 topology를 종이 위에서 설계하는 단계라면, 다음 단위는 그 설계가 실제 runtime에서 어떤 로그·metric·failure mode로 나타나는지 읽는 단계다. 따라서 여기서 parallel axes를 한 장의 시스템 그림으로 묶어 두면, 다음 단위에서 profiling과 recovery를 볼 때도 "이 병목이 tensor parallel 때문인지, pipeline bubble 때문인지, FSDP shard lifecycle 때문인지"를 더 정확히 분해할 수 있다.
