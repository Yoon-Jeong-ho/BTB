from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _lesson_metadata import load_lesson_metadata


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_OUTPUT_PATHS = {
    "scratch metrics json": (Path("artifacts") / "scratch-manual" / "metrics.json",),
    "framework metrics json": (Path("artifacts") / "framework-manual" / "metrics.json",),
    "analysis markdown": (Path("artifacts") / "analysis-manual" / "latest_report.md",),
    "stable analysis markdown": (Path("analysis.md"),),
    "observed analysis report": (Path("artifacts") / "analysis-manual" / "latest_report.md",),
    "observed analysis markdown": (Path("artifacts") / "analysis-manual" / "latest_report.md",),
    "reflection markdown": (Path("reflection.md"),),
    "reflection worksheet": (Path("reflection.md"),),
    "runnable readme": (Path("README.md"),),
    "theory note": (Path("THEORY.md"),),
    "prerequisite checklist": (Path("PREREQS.md"),),
    "lesson metadata": (Path("lesson.yaml"),),
}
PATH_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".png", ".svg", ".txt"}


def _read_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"metric JSON must contain an object: {_to_display(path)}")
    return payload


def _read_metric_keys(path: Path) -> list[str]:
    return sorted(_read_metrics(path).keys())


def _to_display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _standard_patterns(output_name: str) -> tuple[Path, ...] | None:
    normalized = " ".join(output_name.strip().lower().split())
    exact = REQUIRED_OUTPUT_PATHS.get(normalized)
    if exact is not None:
        return exact
    if "scratch" in normalized and "metrics" in normalized and "json" in normalized:
        return (Path("artifacts") / "scratch-manual" / "metrics.json",)
    if "framework" in normalized and "metrics" in normalized and "json" in normalized:
        return (Path("artifacts") / "framework-manual" / "metrics.json",)
    if "scratch" in normalized and "svg" in normalized:
        return (Path("artifacts") / "scratch-manual" / "*.svg",)
    if "framework" in normalized and "svg" in normalized:
        return (Path("artifacts") / "framework-manual" / "*.svg",)
    if "observed" in normalized and "analysis" in normalized and "json" in normalized:
        return (Path("artifacts") / "analysis-manual" / "*.json",)
    if "observed" in normalized and "analysis" in normalized:
        return (Path("artifacts") / "analysis-manual" / "latest_report.md",)
    return None


def _path_like_pattern(output_name: str) -> Path | None:
    candidate = Path(output_name.strip())
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    if "/" in output_name or candidate.suffix.lower() in PATH_SUFFIXES:
        return candidate
    return None


def _resolve_pattern(unit_path: Path, pattern: Path) -> tuple[list[Path], Path]:
    pattern_text = str(pattern)
    expected = unit_path / pattern
    if any(character in pattern_text for character in "*?["):
        return sorted(path for path in unit_path.glob(pattern_text) if path.is_file()), expected
    return ([expected] if expected.is_file() else []), expected


def _resolve_required_outputs(
    unit_path: Path,
    required_outputs: object,
) -> tuple[list[tuple[str, list[Path]]], list[str], list[Path]]:
    if not isinstance(required_outputs, list):
        return [], [], []

    resolved: list[tuple[str, list[Path]]] = []
    unverified: list[str] = []
    missing: list[Path] = []
    for output_name in required_outputs:
        if not isinstance(output_name, str):
            unverified.append(repr(output_name))
            continue
        path_pattern = _path_like_pattern(output_name)
        patterns = (path_pattern,) if path_pattern is not None else _standard_patterns(output_name)
        if patterns is None:
            unverified.append(output_name)
            continue

        matched_paths: list[Path] = []
        for pattern in patterns:
            matches, expected = _resolve_pattern(unit_path, pattern)
            if not matches:
                missing.append(expected)
            matched_paths.extend(matches)
        resolved.append((output_name, matched_paths))
    return resolved, unverified, missing


def _resolve_required_output_paths(unit_path: Path, required_outputs: object) -> list[Path]:
    resolved, _unverified, missing = _resolve_required_outputs(unit_path, required_outputs)
    return [path for _name, paths in resolved for path in paths] + missing


def _ensure_required_outputs_exist(
    unit_path: Path,
    required_outputs: object,
) -> tuple[list[tuple[str, list[Path]]], list[str]]:
    resolved, unverified, missing_paths = _resolve_required_outputs(unit_path, required_outputs)
    if not missing_paths:
        return resolved, unverified

    missing_lines = "\n".join(f"- {_to_display(path)}" for path in missing_paths)
    raise SystemExit(
        "필수 출력이 없습니다. 아래 경로를 먼저 생성하세요:\n"
        f"{missing_lines}\n"
        "먼저 필요한 scratch/framework mode와 analysis.py를 실행해 결과를 만드세요. "
        "전체 순차 실행은 run_lesson.py --mode all을 사용할 수 있습니다."
    )


def _metric_lines(label: str, metrics: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            rendered = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
            lines.append(f"- {label} {key}: {rendered}")
        elif isinstance(value, list) and len(value) <= 8:
            lines.append(f"- {label} {key}: {json.dumps(value, ensure_ascii=False)}")
        if len(lines) >= 10:
            break
    return lines


def _artifact_link(path: Path, summary_path: Path) -> str:
    try:
        href = path.relative_to(summary_path.parent)
    except ValueError:
        href = Path("..") / path.relative_to(summary_path.parent.parent)
    return f"[{_to_display(path)}]({href.as_posix()})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an evidence-linked lesson summary.")
    parser.add_argument("--unit", required=True, help="Unit path, e.g. 00_foundations/01_tensor_shapes")
    args = parser.parse_args()

    unit_path = Path(args.unit)
    if not unit_path.is_absolute():
        unit_path = (ROOT / unit_path).resolve()

    metadata = load_lesson_metadata(unit_path / "lesson.yaml")
    artifacts_dir = unit_path / "artifacts"
    summary_path = artifacts_dir / "summary.md"
    scratch_metrics_path = artifacts_dir / "scratch-manual" / "metrics.json"
    framework_metrics_path = artifacts_dir / "framework-manual" / "metrics.json"
    required_outputs = metadata.get("required_outputs", [])
    resolved_outputs, unverified = _ensure_required_outputs_exist(unit_path, required_outputs)
    scratch_metrics = _read_metrics(scratch_metrics_path)
    framework_metrics = _read_metrics(framework_metrics_path)

    summary_lines = [
        f"# {unit_path.name} 요약",
        "",
        "## 목적",
        f"- {metadata.get('objective', '')}",
        "",
        "## 출력 스캐폴드",
        f"- scratch keys: {sorted(scratch_metrics)}",
        f"- framework keys: {sorted(framework_metrics)}",
    ]
    summary_lines.extend(_metric_lines("scratch", scratch_metrics))
    summary_lines.extend(_metric_lines("framework", framework_metrics))

    summary_lines.extend(["", "## Artifact evidence"])
    linked_paths: set[Path] = set()
    for declaration, paths in resolved_outputs:
        for path in paths:
            if path in linked_paths:
                continue
            linked_paths.add(path)
            summary_lines.append(f"- {declaration}: {_artifact_link(path, summary_path)}")
    if not linked_paths:
        summary_lines.append("- concretely resolved artifact declarations: none")

    summary_lines.extend(["", "## unverified declarations"])
    if unverified:
        summary_lines.extend(f"- {declaration}" for declaration in unverified)
    else:
        summary_lines.append("- none")

    questions = metadata.get("analysis_questions", [])
    summary_lines.extend(["", "## 분석 질문"])
    if isinstance(questions, list) and questions:
        summary_lines.extend(f"- {question}" for question in questions)
    else:
        summary_lines.append("- analysis.md에서 왜 이런 결과가 나왔는지 설명하기")
    summary_lines.extend(
        [
            "",
            "## 다음 질문",
            "- reflection.md에 이번 결과의 한계와 다음 실험에서 바꿀 점 적기",
            "",
        ]
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(_to_display(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
