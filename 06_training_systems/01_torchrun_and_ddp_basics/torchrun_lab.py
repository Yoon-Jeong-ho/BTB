from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


ARTIFACT = Path(__file__).parent / "artifacts" / "torchrun-manual" / "metrics.json"


def main() -> None:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise SystemExit(
            "torchrun environment is missing. Run with: python -m torch.distributed.run "
            "--standalone --nproc-per-node=2 06_training_systems/01_torchrun_and_ddp_basics/torchrun_lab.py"
        )
    if os.environ.get("BTB_DEVICE", "cpu").strip().lower() != "cpu":
        raise SystemExit("This introductory torchrun milestone uses BTB_DEVICE=cpu and the gloo backend.")

    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_value = torch.tensor(float(rank + 1))
        dist.all_reduce(local_value, op=dist.ReduceOp.SUM)
        local_value /= world_size

        observed_ranks: list[int | None] = [None] * world_size
        dist.all_gather_object(observed_ranks, rank)
        dist.barrier()
        if rank == 0:
            payload = {
                "framework": "torch.distributed",
                "backend": dist.get_backend(),
                "device": "cpu",
                "world_size": world_size,
                "observed_ranks": observed_ranks,
                "all_reduce_mean": local_value.item(),
            }
            ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
            ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
