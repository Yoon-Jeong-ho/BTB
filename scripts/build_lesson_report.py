from __future__ import annotations

import argparse
import json
from pathlib import Path

from _lesson_metadata import load_lesson_metadata


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_OUTPUT_PATHS = {
    "scratch metrics json": Path("artifacts") / "scratch-manual" / "metrics.json",
    "framework metrics json": Path("artifacts") / "framework-manual" / "metrics.json",
    "analysis markdown": Path("analysis.md"),
}


def _read_metric_keys(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sorted(payload.keys())


def _to_display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _resolve_required_output_paths(unit_path: Path, required_outputs: object) -> list[Path]:
    if not isinstance(required_outputs, list):
        return []

    resolved_paths: list[Path] = []
    for output_name in required_outputs:
        if not isinstance(output_name, str):
            continue
        relative_path = REQUIRED_OUTPUT_PATHS.get(output_name.strip().lower())
        if relative_path is None:
            continue
        resolved_paths.append(unit_path / relative_path)
    return resolved_paths


def _ensure_required_outputs_exist(unit_path: Path, required_outputs: object) -> None:
    expected_paths = _resolve_required_output_paths(unit_path, required_outputs)
    missing_paths = [path for path in expected_paths if not path.exists()]
    if not missing_paths:
        return

    missing_lines = "\n".join(f"- {_to_display(path)}" for path in missing_paths)
    raise SystemExit(
        "필수 출력이 없습니다. 아래 경로를 먼저 생성하세요:\n"
        f"{missing_lines}\n"
        "먼저 scratch_lab.py, framework_lab.py, analysis.py를 순서대로 실행해 결과를 만드세요."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a stable lesson summary scaffold.")
    parser.add_argument("--unit", required=True, help="Unit path, e.g. 00_foundations/01_tensor_shapes")
    args = parser.parse_args()

    unit_path = Path(args.unit)
    if not unit_path.is_absolute():
        unit_path = (ROOT / unit_path).resolve()

    metadata = load_lesson_metadata(unit_path / "lesson.yaml")
    artifacts_dir = unit_path / "artifacts"
    summary_path = artifacts_dir / "summary.md"
    scratch_metrics = artifacts_dir / "scratch-manual" / "metrics.json"
    framework_metrics = artifacts_dir / "framework-manual" / "metrics.json"
    required_outputs = metadata.get("required_outputs", [])
    _ensure_required_outputs_exist(unit_path, required_outputs)

    summary_lines = [
        f"# {unit_path.name} 요약",
        "",
        "## 목적",
        f"- {metadata.get('objective', '')}",
        "",
        "## 출력 스캐폴드",
        f"- scratch keys: {_read_metric_keys(scratch_metrics)}",
        f"- framework keys: {_read_metric_keys(framework_metrics)}",
    ]

    if isinstance(required_outputs, list):
        summary_lines.append(f"- required outputs: {required_outputs}")

    summary_lines.extend(
        [
            "",
            "## 다음 질문",
            "- analysis.md에서 왜 이런 결과가 나왔는지 설명하기",
            "- reflection.md에 다음 실험에서 바꿀 점 적기",
            "",
        ]
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(_to_display(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
