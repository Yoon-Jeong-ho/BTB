from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from _lesson_metadata import LessonValue, load_lesson_metadata


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FIELDS = (
    "objective",
    "prereqs",
    "key_terms",
    "required_outputs",
    "analysis_questions",
    "fidelity",
    "difficulty",
    "estimated_minutes",
    "compute",
)
LIST_FIELDS = ("prereqs", "key_terms", "required_outputs", "analysis_questions")
ALLOWED_VALUES = {
    "fidelity": {"concept-toy", "framework-toy", "real-data", "gpu-capable"},
    "difficulty": {"beginner", "intermediate", "advanced"},
    "compute": {"cpu", "cpu-or-cuda", "optional-multiprocess"},
}
PREREQUISITE_REFERENCE_RE = re.compile(
    r"^(?P<path>(?:\d{2}_[A-Za-z0-9_]+/\d{2}_[A-Za-z0-9_]+|docs/[A-Za-z0-9_./-]+))"
)


def _display(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _validate_metadata(root: Path, path: Path, metadata: dict[str, LessonValue]) -> list[str]:
    errors: list[str] = []
    display = _display(root, path)
    for key in REQUIRED_FIELDS:
        if key not in metadata:
            errors.append(f"{display}: missing required metadata field '{key}'")

    objective = metadata.get("objective")
    if objective is not None and (not isinstance(objective, str) or not objective.strip()):
        errors.append(f"{display}: objective must be a non-empty string")

    for key in LIST_FIELDS:
        value = metadata.get(key)
        if value is not None and not isinstance(value, list):
            errors.append(f"{display}: {key} must be a list")

    for key, allowed in ALLOWED_VALUES.items():
        value = metadata.get(key)
        if value is not None and (not isinstance(value, str) or value not in allowed):
            errors.append(f"{display}: {key} must be one of {sorted(allowed)}")

    minutes = metadata.get("estimated_minutes")
    if minutes is not None:
        try:
            valid_minutes = int(minutes) > 0
        except (TypeError, ValueError):
            valid_minutes = False
        if not valid_minutes:
            errors.append(f"{display}: estimated_minutes must be a positive integer")
    return errors


def _declared_scripts(metadata: dict[str, LessonValue]) -> list[str]:
    scripts = metadata.get("scripts")
    if isinstance(scripts, list):
        return scripts
    if isinstance(scripts, dict):
        return list(scripts.values())
    return []


def _validate_prerequisite_references(
    root: Path,
    lesson_path: Path,
    metadata: dict[str, LessonValue],
) -> list[str]:
    errors: list[str] = []
    prereqs = metadata.get("prereqs")
    if not isinstance(prereqs, list):
        return errors
    for declaration in prereqs:
        match = PREREQUISITE_REFERENCE_RE.match(str(declaration).strip())
        if not match:
            continue
        reference = match.group("path").rstrip("./")
        candidate = root / reference
        if reference.startswith("docs/") and not candidate.suffix:
            candidate = candidate.with_suffix(".md")
        if not candidate.exists():
            errors.append(
                f"{_display(root, lesson_path)}: prerequisite reference does not exist: {reference}"
            )
    return errors


def _validate_resources(
    root: Path,
    unit_path: Path,
    status: str,
    metadata: dict[str, LessonValue],
) -> list[str]:
    errors: list[str] = []
    for filename in ("README.md", "lesson.yaml"):
        path = unit_path / filename
        if not path.is_file():
            errors.append(f"{_display(root, path)}: required lesson resource is missing")

    for filename in _declared_scripts(metadata):
        path = unit_path / filename
        if not path.is_file():
            errors.append(f"{_display(root, path)}: declared script is missing")

    if status == "runnable":
        stage_entrypoint = unit_path / "run_stage.py"
        standard_entrypoints = [unit_path / name for name in ("scratch_lab.py", "framework_lab.py", "analysis.py")]
        if not stage_entrypoint.is_file() and not all(path.is_file() for path in standard_entrypoints):
            missing = [path.name for path in standard_entrypoints if not path.is_file()]
            errors.append(
                f"{_display(root, unit_path)}: runnable unit needs run_stage.py or scratch/framework/analysis "
                f"entrypoints (missing: {', '.join(missing)})"
            )
    return errors


def audit_curriculum(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = root / "docs" / "curriculum_status.json"
    errors: list[str] = []
    coverage: dict[str, Counter[str]] = {
        "status": Counter(),
        "fidelity": Counter(),
        "difficulty": Counter(),
        "compute": Counter(),
        "runtime": Counter(),
    }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tracks = manifest["tracks"]
        if not isinstance(tracks, dict):
            raise TypeError("'tracks' must be an object")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {
            "unit_count": 0,
            "errors": [f"{_display(root, manifest_path)}: invalid manifest: {exc}"],
            "coverage": {key: {} for key in coverage},
        }

    unit_count = 0
    for track_name, units in tracks.items():
        if not isinstance(units, dict):
            errors.append(f"docs/curriculum_status.json: track '{track_name}' must map units to statuses")
            continue
        for unit_name, status in units.items():
            unit_count += 1
            coverage["status"][str(status)] += 1
            unit_path = root / str(track_name) / str(unit_name)
            lesson_path = unit_path / "lesson.yaml"
            try:
                metadata = load_lesson_metadata(lesson_path)
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
                continue

            errors.extend(_validate_metadata(root, lesson_path, metadata))
            errors.extend(_validate_prerequisite_references(root, lesson_path, metadata))
            errors.extend(_validate_resources(root, unit_path, str(status), metadata))
            declared_status = metadata.get("status")
            if declared_status is not None and declared_status != status:
                errors.append(
                    f"{_display(root, lesson_path)}: metadata status '{declared_status}' "
                    f"does not match manifest status '{status}'"
                )
            for key in ("fidelity", "difficulty", "compute"):
                value = metadata.get(key, "missing")
                coverage[key][str(value)] += 1
            runtime = metadata.get("runtime") or metadata.get("compute") or "missing"
            coverage["runtime"][str(runtime)] += 1

    return {
        "unit_count": unit_count,
        "errors": errors,
        "coverage": {
            key: dict(sorted(counter.items()))
            for key, counter in coverage.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit every lesson declared by the curriculum manifest.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--strict", action="store_true", help="Return non-zero when any audit error is found.")
    args = parser.parse_args()

    result = audit_curriculum(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.strict and result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
