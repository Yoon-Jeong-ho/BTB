from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


UNIT = "vision_language_action_grounding"
ROOT = Path(__file__).resolve().parent
SCRATCH_METRICS = ROOT / "artifacts" / "scratch-manual" / "metrics.json"
FRAMEWORK_METRICS = ROOT / "artifacts" / "framework-manual" / "metrics.json"
ANALYSIS_DIR = ROOT / "artifacts" / "analysis-manual"
REPORT = ANALYSIS_DIR / "latest_report.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _require_keys(name: str, payload: dict[str, Any], keys: set[str]) -> None:
    missing = sorted(keys - set(payload))
    if missing:
        raise ValueError(f"{name} metrics missing keys: {', '.join(missing)}")


def write_report(scratch: dict[str, Any], framework: dict[str, Any]) -> dict[str, Any]:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "unit": UNIT,
        "status": "runnable",
        "scratch_action_accuracy": scratch["action_accuracy"],
        "scratch_safety_gate_accuracy": scratch["safety_gate_accuracy"],
        "framework_action_accuracy": framework["action_accuracy"],
        "framework_safety_gate_accuracy": framework["safety_gate_accuracy"],
        "framework_device": framework["device"],
    }
    report = f"""# 01 Vision-Language-Action Grounding 실행 관측

## 관측 요약

- scratch action accuracy: {scratch['action_accuracy']}
- scratch safety gate accuracy: {scratch['safety_gate_accuracy']}
- framework action accuracy: {framework['action_accuracy']}
- framework safety gate accuracy: {framework['safety_gate_accuracy']}
- framework device: {framework['device']}

## 한국어 해석

이 toy VLA 단위는 이미지-텍스트 이해를 `action token` 선택으로 확장한다. scratch policy matrix와 tiny framework policy head 모두 정답 action을 고르지만, 실제 VLA에서는 action만 맞는 것으로 충분하지 않다. 위험 장면에서 실행을 막는 `safety gate`가 별도 지표로 남아야 한다.

## 다음 실험으로 확장할 로그

- success rate
- trajectory error
- intervention count
- safety violation
- 실패 장면 replay 또는 qualitative panel

## 관련 이론

- [THEORY.md](../../THEORY.md)
- [PREREQS.md](../../PREREQS.md)
"""
    REPORT.write_text(report, encoding="utf-8")
    return payload


def main() -> int:
    try:
        scratch = _load_json(SCRATCH_METRICS)
        framework = _load_json(FRAMEWORK_METRICS)
        _require_keys("scratch", scratch, {"action_accuracy", "safety_gate_accuracy", "policy_matrix_shape"})
        _require_keys("framework", framework, {"action_accuracy", "safety_gate_accuracy", "logits_shape", "device"})
    except FileNotFoundError as exc:
        print(
            f"필수 VLA metrics 파일이 없습니다: {exc}. 먼저 scratch_lab.py와 framework_lab.py를 실행하세요.",
            file=sys.stderr,
        )
        return 1
    except ValueError as exc:
        print(f"VLA metrics schema validation failed: {exc}", file=sys.stderr)
        return 1

    payload = write_report(scratch, framework)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
