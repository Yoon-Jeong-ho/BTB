from __future__ import annotations

import os

import torch


VALID_DEVICE_REQUESTS = {"auto", "cpu", "cuda"}


def resolve_torch_device(requested: str | None = None) -> torch.device:
    """Resolve BTB's small-lab device contract without silently faking CUDA."""

    # Direct lesson execution stays CPU-safe; runners opt into auto explicitly.
    value = (requested or os.environ.get("BTB_DEVICE", "cpu")).strip().lower()
    if value not in VALID_DEVICE_REQUESTS:
        raise ValueError("BTB_DEVICE must be auto, cpu, or cuda")
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("BTB_DEVICE=cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
