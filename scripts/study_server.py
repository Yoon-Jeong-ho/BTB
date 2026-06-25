from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
RUNNABLE_NAMES = {"scratch_lab.py", "framework_lab.py", "analysis.py", "run_stage.py"}
DEFAULT_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class RunnerConfig:
    python: str | None = None
    conda_env: str | None = None
    conda_prefix: str | None = None
    device: str = "auto"
    gpu_index: str | None = None
    gpu_max_used_mb: int = 2048
    gpu_max_util_percent: int = 25


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _resolve_runnable_path(raw_path: str) -> Path:
    cleaned = unquote(str(raw_path or "")).strip()
    if cleaned.startswith("/"):
        cleaned = cleaned.lstrip("/")
    while cleaned.startswith("../"):
        cleaned = cleaned[3:]

    candidate = (ROOT / cleaned).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError as exc:
        raise PermissionError("repository 밖의 파일은 실행할 수 없습니다.") from exc

    if candidate.name not in RUNNABLE_NAMES:
        raise PermissionError("scratch_lab.py, framework_lab.py, analysis.py, run_stage.py만 실행할 수 있습니다.")
    if not candidate.is_file():
        raise FileNotFoundError(cleaned)
    return candidate


def _parse_nvidia_smi_rows(output: str) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            rows.append(
                {
                    "index": parts[0],
                    "memory_free_mb": int(parts[1]),
                    "memory_used_mb": int(parts[2]),
                    "utilization_percent": int(parts[3]),
                }
            )
        except ValueError:
            continue
    return rows


def _query_gpu_rows() -> list[dict[str, int | str]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    return _parse_nvidia_smi_rows(completed.stdout)


def _select_idle_gpu(
    gpu_rows: list[dict[str, int | str]],
    *,
    gpu_index: str | None = None,
    max_used_mb: int = 2048,
    max_util_percent: int = 25,
    require_idle: bool = True,
) -> dict[str, int | str] | None:
    if gpu_index is not None:
        return next((row for row in gpu_rows if str(row["index"]) == str(gpu_index)), None)

    candidates = [
        row
        for row in gpu_rows
        if int(row["memory_used_mb"]) <= max_used_mb and int(row["utilization_percent"]) <= max_util_percent
    ]
    if not candidates and not require_idle:
        candidates = gpu_rows
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (-int(row["memory_free_mb"]), int(row["utilization_percent"]), int(row["memory_used_mb"]), str(row["index"])),
    )[0]


def _python_prefix(config: RunnerConfig) -> tuple[list[str], str]:
    if config.conda_env:
        return ["conda", "run", "--no-capture-output", "-n", config.conda_env, "python"], f"conda:{config.conda_env}"
    if config.conda_prefix:
        return ["conda", "run", "--no-capture-output", "-p", config.conda_prefix, "python"], f"conda-prefix:{config.conda_prefix}"
    python = config.python or sys.executable
    return [python], f"python:{python}"


def _build_runner_invocation(
    script_path: Path,
    config: RunnerConfig,
    gpu_rows: list[dict[str, int | str]] | None = None,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    rows = _query_gpu_rows() if gpu_rows is None else gpu_rows
    device = config.device
    selected_gpu: dict[str, int | str] | None = None
    reason = ""

    if device == "cpu":
        reason = "CPU mode requested."
    elif device == "cuda":
        selected_gpu = _select_idle_gpu(
            rows,
            gpu_index=config.gpu_index,
            max_used_mb=config.gpu_max_used_mb,
            max_util_percent=config.gpu_max_util_percent,
            require_idle=False,
        )
        reason = "CUDA mode requested."
    else:
        selected_gpu = _select_idle_gpu(
            rows,
            gpu_index=config.gpu_index,
            max_used_mb=config.gpu_max_used_mb,
            max_util_percent=config.gpu_max_util_percent,
            require_idle=True,
        )
        if selected_gpu is None:
            device = "cpu"
            reason = "No idle GPU matched the threshold; falling back to CPU."
        else:
            device = "cuda"
            reason = "Idle GPU selected automatically."

    command_prefix, environment_label = _python_prefix(config)
    relative_script = str(script_path.relative_to(ROOT))
    env_overlay = {"BTB_DEVICE": device}
    if device == "cpu":
        env_overlay["CUDA_VISIBLE_DEVICES"] = ""
    elif selected_gpu is not None:
        env_overlay["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["index"])

    runner = {
        "python": " ".join(command_prefix),
        "environment": environment_label,
        "device": device,
        "gpu_index": str(selected_gpu["index"]) if selected_gpu is not None else None,
        "gpu": selected_gpu,
        "device_reason": reason,
        "gpu_policy": {
            "max_used_mb": config.gpu_max_used_mb,
            "max_util_percent": config.gpu_max_util_percent,
        },
    }
    return command_prefix + [relative_script], env_overlay, runner


class StudyRequestHandler(SimpleHTTPRequestHandler):
    server_version = "BTBStudyServer/1.0"

    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, directory=directory or str(ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if self.path != "/api/run-python":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            request = json.loads(body or "{}")
            script_path = _resolve_runnable_path(str(request.get("path", "")))
            timeout = min(max(int(request.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)), 1), DEFAULT_TIMEOUT_SECONDS)
        except PermissionError as exc:
            self._send_json(HTTPStatus.FORBIDDEN, {"error": str(exc)})
            return
        except FileNotFoundError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"파일을 찾을 수 없습니다: {exc}"})
            return
        except Exception as exc:  # pragma: no cover - malformed client request
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"요청 형식이 올바르지 않습니다: {exc}"})
            return

        started = time.monotonic()
        config: RunnerConfig = getattr(self.server, "runner_config", RunnerConfig())  # type: ignore[attr-defined]
        command, env_overlay, runner = _build_runner_invocation(script_path, config)
        try:
            env = os.environ.copy()
            env.update(env_overlay)
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
            status = HTTPStatus.OK
            payload = {
                "path": str(script_path.relative_to(ROOT)),
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "duration_seconds": round(time.monotonic() - started, 3),
                "runner": runner,
            }
        except FileNotFoundError as exc:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            payload = {
                "path": str(script_path.relative_to(ROOT)),
                "command": command,
                "returncode": 127,
                "stdout": "",
                "stderr": f"실행기를 찾을 수 없습니다: {exc.filename}",
                "duration_seconds": round(time.monotonic() - started, 3),
                "runner": runner,
            }
        except subprocess.TimeoutExpired as exc:
            status = HTTPStatus.REQUEST_TIMEOUT
            payload = {
                "path": str(script_path.relative_to(ROOT)),
                "command": command,
                "returncode": 124,
                "stdout": exc.stdout or "",
                "stderr": f"실행 시간이 {timeout}초를 넘었습니다.",
                "duration_seconds": round(time.monotonic() - started, 3),
                "runner": runner,
            }

        self._send_json(status, payload)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        data = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the BTB study website with a local allowlisted Python runner.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--python", dest="python", default=None, help="Python executable to use when not using conda.")
    parser.add_argument("--conda-env", default=None, help="Run labs through `conda run -n ENV python ...`.")
    parser.add_argument("--conda-prefix", default=None, help="Run labs through `conda run -p PREFIX python ...`.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="auto picks an idle GPU and falls back to CPU.")
    parser.add_argument("--gpu-index", default=None, help="Prefer this GPU index when --device auto/cuda is used.")
    parser.add_argument("--gpu-max-used-mb", type=int, default=2048, help="Auto GPU is considered idle below this used-memory threshold.")
    parser.add_argument("--gpu-max-util-percent", type=int, default=25, help="Auto GPU is considered idle below this utilization threshold.")
    args = parser.parse_args()
    if args.conda_env and args.conda_prefix:
        parser.error("--conda-env and --conda-prefix are mutually exclusive")

    runner_config = RunnerConfig(
        python=args.python,
        conda_env=args.conda_env,
        conda_prefix=args.conda_prefix,
        device=args.device,
        gpu_index=args.gpu_index,
        gpu_max_used_mb=args.gpu_max_used_mb,
        gpu_max_util_percent=args.gpu_max_util_percent,
    )

    server = ThreadingHTTPServer((args.bind, args.port), StudyRequestHandler)
    server.runner_config = runner_config  # type: ignore[attr-defined]
    print(f"BTB study server: http://{args.bind}:{args.port}/web/")
    print(
        "Runner: "
        f"environment={_python_prefix(runner_config)[1]}, "
        f"device={runner_config.device}, "
        f"gpu_index={runner_config.gpu_index or 'auto'}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
