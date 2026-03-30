from __future__ import annotations

import argparse
import runpy
from pathlib import Path

from _lesson_metadata import load_lesson_metadata


ROOT = Path(__file__).resolve().parents[1]
MODE_TO_SCRIPT = {
    "scratch": "scratch_lab.py",
    "framework": "framework_lab.py",
}
MODE_TO_ARTIFACT_DIR = {
    "scratch": "scratch-manual",
    "framework": "framework-manual",
}


def _resolve_unit(unit_arg: str) -> tuple[Path, str]:
    requested = Path(unit_arg)
    unit_path = requested if requested.is_absolute() else ROOT / requested
    unit_path = unit_path.resolve()

    try:
        display = str(unit_path.relative_to(ROOT))
    except ValueError:
        display = str(unit_path)

    return unit_path, display


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a foundation lesson unit.")
    parser.add_argument("--unit", required=True, help="Unit path, e.g. 00_foundations/01_tensor_shapes")
    parser.add_argument("--mode", choices=sorted(MODE_TO_SCRIPT), required=True)
    args = parser.parse_args()

    unit_path, display_unit = _resolve_unit(args.unit)
    lesson_path = unit_path / "lesson.yaml"
    target_path = unit_path / MODE_TO_SCRIPT[args.mode]

    if not lesson_path.exists():
        raise SystemExit(f"lesson metadata not found: {lesson_path}")
    if not target_path.exists():
        raise SystemExit(f"lesson entrypoint not found: {target_path}")

    metadata = load_lesson_metadata(lesson_path)
    runpy.run_path(str(target_path), run_name="__main__")

    artifact_dir = unit_path / "artifacts" / MODE_TO_ARTIFACT_DIR[args.mode]
    try:
        display_artifact_dir = artifact_dir.relative_to(ROOT)
    except ValueError:
        display_artifact_dir = artifact_dir

    print(
        " ".join(
            [
                f"unit={display_unit}",
                f"mode={args.mode}",
                f"objective={metadata.get('objective', '')}",
                f"artifact_dir={display_artifact_dir}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
