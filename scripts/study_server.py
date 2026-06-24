from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
RUNNABLE_NAMES = {"scratch_lab.py", "framework_lab.py", "analysis.py"}
DEFAULT_TIMEOUT_SECONDS = 60


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
        raise PermissionError("scratch_lab.py, framework_lab.py, analysis.py만 실행할 수 있습니다.")
    if not candidate.is_file():
        raise FileNotFoundError(cleaned)
    return candidate


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
        try:
            completed = subprocess.run(
                [sys.executable, str(script_path.relative_to(ROOT))],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
            status = HTTPStatus.OK
            payload = {
                "path": str(script_path.relative_to(ROOT)),
                "command": [sys.executable, str(script_path.relative_to(ROOT))],
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "duration_seconds": round(time.monotonic() - started, 3),
            }
        except subprocess.TimeoutExpired as exc:
            status = HTTPStatus.REQUEST_TIMEOUT
            payload = {
                "path": str(script_path.relative_to(ROOT)),
                "returncode": 124,
                "stdout": exc.stdout or "",
                "stderr": f"실행 시간이 {timeout}초를 넘었습니다.",
                "duration_seconds": round(time.monotonic() - started, 3),
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
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.bind, args.port), StudyRequestHandler)
    print(f"BTB study server: http://{args.bind}:{args.port}/web/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
