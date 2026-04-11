from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
OBSERVED = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding='utf-8'))


def ensure_metrics() -> None:
    missing = [path for path in [SCRATCH, FRAMEWORK] if not path.exists()]
    if missing:
        names = ', '.join(str(path.relative_to(UNIT_ROOT)) for path in missing)
        raise SystemExit(
            f'필수 metrics 파일이 없습니다: {names}. '
            '먼저 scratch_lab.py와 framework_lab.py를 실행하세요. '
            '예: python3 06_training_systems/04_fsdp_checkpointing_and_offload/scratch_lab.py && '
            'python3 06_training_systems/04_fsdp_checkpointing_and_offload/framework_lab.py'
        )


def run() -> None:
    ensure_metrics()
    scratch = load(SCRATCH)
    framework = load(FRAMEWORK)
    full_state = framework['state_dict_modes']['full_state_dict']
    sharded_state = framework['state_dict_modes']['sharded_state_dict']
    no_offload = framework['offload_policy']['none']
    offload = framework['offload_policy']['cpu_optimizer_offload']
    report = f'''# 04 FSDP Checkpointing and Offload 실행 관측

## 관측 결과
- DDP full replica per-rank memory: `{scratch['ddp_full_replica_per_rank_mb']} MB`
- FSDP forward peak without activation checkpointing: `{scratch['fsdp_forward_peak_gpu_mb']} MB`
- FSDP peak with activation checkpointing: `{scratch['fsdp_checkpointed_peak_gpu_mb']} MB`
- CPU optimizer offload peak: `{scratch['cpu_offload_gpu_peak_mb']} MB`
- activation checkpoint recompute multiplier: `{scratch['checkpoint_recompute_multiplier']}x`
- no-offload simulated step time: `{no_offload['step_time_ms']} ms`
- CPU offload simulated step time: `{offload['step_time_ms']} ms`
- full state dict load peak: `{full_state['load_peak_mb']} MB`
- sharded state dict load peak: `{sharded_state['load_peak_mb']} MB`

## 한국어 해석
- FSDP는 기본 거주 상태를 shard로 두고, forward 직전에 `all_gather_full_params`로 full view를 잠깐 만든 뒤 다시 버리는 lifecycle로 읽는다.
- activation checkpointing은 activation 저장량을 줄여 peak memory를 낮추지만, `{scratch['checkpoint_recompute_multiplier']}x` 재계산 비용을 step time에 남긴다.
- CPU offload는 GPU peak를 더 낮추지만 `{offload['transfer_ms']} ms` 전송 비용을 추가하므로, 속도 최적화가 아니라 메모리 생존성 선택지로 해석해야 한다.
- full state dict는 export/debug에는 단순하지만 load peak가 크고, sharded state dict는 resume memory가 낮지만 world size 변경 또는 다른 runtime 이식 시 merge/reshard 계약이 필요하다.

## 실행 조치
- OOM이 forward 직전 peak에서 난다면 auto-wrap granularity와 activation checkpointing부터 줄여 본다.
- optimizer state가 GPU peak를 밀어 올린다면 CPU offload를 켜되 step-time 회귀를 별도 metric으로 기록한다.
- preemption이 잦은 학습은 sharded state dict resume을 기본으로 두고, 릴리스/export 시점에만 full state dict를 만든다.
'''
    OBSERVED.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED.write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    run()
