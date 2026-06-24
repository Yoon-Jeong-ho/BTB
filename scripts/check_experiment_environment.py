from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return f"unavailable: {exc}"


def _module_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "installed"))
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"missing: {exc.__class__.__name__}"


def gpu_snapshot() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    output = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, total, free, util = parts
        gpus.append({
            "index": int(index),
            "name": name,
            "memory_total_mib": int(total),
            "memory_free_mib": int(free),
            "utilization_gpu_percent": int(util),
            "recommended_for_optional_heavy_lab": int(free) >= 24000 and int(util) <= 10,
        })
    return gpus


def snapshot() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "packages": {
            "numpy": _module_version("numpy"),
            "torch": _module_version("torch"),
            "sklearn": _module_version("sklearn"),
            "yaml": _module_version("yaml"),
        },
        "gpus": gpu_snapshot(),
    }


def main() -> int:
    payload = snapshot()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
