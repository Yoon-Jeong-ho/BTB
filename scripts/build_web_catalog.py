from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from _lesson_metadata import LessonValue, load_lesson_metadata

HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def _title_from_readme(path: Path, fallback_id: str) -> str:
    if path.exists():
        text = path.read_text(encoding="utf-8")
        match = HEADING_RE.search(text)
        if match:
            return match.group(1).strip()
    return fallback_id.replace("_", " ").title()


def _first_korean_paragraph(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    for paragraph in paragraphs[1:]:
        if paragraph.startswith("#") or paragraph.startswith("|") or paragraph.startswith("```"):
            continue
        return " ".join(paragraph.split())
    return ""


def _as_list(value: LessonValue | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _as_positive_int(value: LessonValue | None) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def _existing_checkpoints(unit_path: Path) -> list[str]:
    return [resource["checkpoint"] for resource in _unit_resources(unit_path) if resource.get("checkpoint")]


def _resource_type(filename: str) -> str:
    return "markdown" if filename.endswith(".md") else "code"


def _resource_entry(unit_path: Path, filename: str, label: str | None = None, checkpoint: str | None = None) -> dict[str, str]:
    return {
        "id": filename.replace(".", "-").replace("_", "-"),
        "label": label or filename,
        "href": f"{unit_path.parent.name}/{unit_path.name}/{filename}",
        "type": _resource_type(filename),
        "language": "python" if filename.endswith(".py") else "markdown",
        "checkpoint": checkpoint or "",
    }


def _is_substantive_ml_helper(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    executable_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and not line.strip().startswith("#")
        and not line.strip().startswith("from __future__")
    ]
    definitions = re.findall(r"^(?:def|class)\s+\w+", text, flags=re.MULTILINE)
    model_or_pipeline_terms = [
        "Pipeline(",
        "Dummy",
        "LogisticRegression",
        "RandomForest",
        "HistGradientBoosting",
        "train_torch",
        "ModelResult",
        "ColumnTransformer",
        "OneHotEncoder",
        "StratifiedShuffleSplit",
    ]
    return len(executable_lines) >= 20 and (len(definitions) >= 2 or any(term in text for term in model_or_pipeline_terms))


def _unit_resources(unit_path: Path) -> list[dict[str, str]]:
    standard = [
        ("README.md", "README", "README"),
        ("THEORY.md", "THEORY", "THEORY"),
        ("PREREQS.md", "PREREQS", "PREREQS"),
        ("scratch_lab.py", "scratch_lab.py", "scratch lab"),
        ("framework_lab.py", "framework_lab.py", "framework lab"),
        # Optional sidecar: expose it without making it part of mandatory mastery checkpoints.
        ("torchrun_lab.py", "torchrun_lab.py", ""),
        ("analysis.py", "analysis.py", "analysis script"),
        ("analysis.md", "analysis.md", "analysis note"),
        ("reflection.md", "reflection.md", "reflection"),
    ]
    ml_runner = [
        ("README.md", "README", "README"),
        ("THEORY.md", "THEORY", "THEORY"),
        ("dataset.py", "dataset.py", "실습 구성"),
        ("models.py", "models.py", ""),
        ("experiment.py", "experiment.py", "실험 실행"),
        ("run_stage.py", "run_stage.py", "실행 명령"),
        ("analysis.py", "analysis.py", "analysis script"),
        ("report.py", "report.py", ""),
    ]
    candidates = ml_runner if (unit_path / "run_stage.py").exists() and not (unit_path / "scratch_lab.py").exists() else standard
    optional_ml_helpers = {"dataset.py", "models.py", "analysis.py", "report.py"}
    return [
        _resource_entry(unit_path, filename, label, checkpoint)
        for filename, label, checkpoint in candidates
        if (unit_path / filename).exists()
        and (candidates is not ml_runner or filename not in optional_ml_helpers or _is_substantive_ml_helper(unit_path / filename))
    ]


def _unit_entry(root: Path, track_id: str, unit_id: str, status: str) -> dict[str, Any]:
    unit_path = root / track_id / unit_id
    lesson_path = unit_path / "lesson.yaml"
    metadata = load_lesson_metadata(lesson_path) if lesson_path.exists() else {}
    readme_path = unit_path / "README.md"

    objective = str(metadata.get("objective") or _first_korean_paragraph(readme_path))
    required_outputs = _as_list(metadata.get("required_outputs"))
    resources = _unit_resources(unit_path)

    return {
        "id": unit_id,
        "title": _title_from_readme(readme_path, unit_id),
        "path": f"{track_id}/{unit_id}",
        "status": status,
        "objective": objective,
        "readme": f"{track_id}/{unit_id}/README.md",
        "lesson": f"{track_id}/{unit_id}/lesson.yaml" if lesson_path.exists() else "",
        "prereqs": _as_list(metadata.get("prereqs")),
        "key_terms": _as_list(metadata.get("key_terms")),
        "required_outputs": required_outputs,
        "analysis_questions": _as_list(metadata.get("analysis_questions")),
        "fidelity": str(metadata.get("fidelity", "")),
        "difficulty": str(metadata.get("difficulty", "")),
        "estimated_minutes": _as_positive_int(metadata.get("estimated_minutes")),
        "compute": str(metadata.get("compute", "")),
        "resources": resources,
        "checkpoints": _existing_checkpoints(unit_path),
        "cpu_safe": str(metadata.get("cpu_safe", "")).lower() == "true",
        "deterministic": str(metadata.get("deterministic", "")).lower() == "true",
    }


def build_catalog(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = root / "docs" / "curriculum_status.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tracks_manifest = manifest["tracks"]

    tracks: list[dict[str, Any]] = []
    for order, (track_id, units) in enumerate(tracks_manifest.items()):
        track_path = root / track_id
        track_readme = track_path / "README.md"
        track_entry = {
            "id": track_id,
            "title": _title_from_readme(track_readme, track_id),
            "order": order,
            "readme": f"{track_id}/README.md",
            "summary": _first_korean_paragraph(track_readme),
            "units": [
                _unit_entry(root, track_id, unit_id, status)
                for unit_id, status in units.items()
            ],
        }
        tracks.append(track_entry)

    return {
        "schema_version": 1,
        "source": "BTB repo curriculum_status.json + lesson.yaml",
        "tracks": tracks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build static catalog for the BTB study website.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", default="web/catalog.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    catalog = build_catalog(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        display_output = output.relative_to(root)
    except ValueError:
        display_output = output
    print(f"wrote {display_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
