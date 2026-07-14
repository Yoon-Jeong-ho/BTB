from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

from _lesson_metadata import LessonValue, load_lesson_metadata


ROOT = Path(__file__).resolve().parents[1]
MODE_ORDER = ("scratch", "framework", "analysis")
MODE_TO_SCRIPT = {
    "scratch": "scratch_lab.py",
    "framework": "framework_lab.py",
    "analysis": "analysis.py",
    "stage": "run_stage.py",
}
MODE_TO_ARTIFACT_DIR = {
    "scratch": "scratch-manual",
    "framework": "framework-manual",
    "analysis": "",
    "stage": "",
}


def _artifact_dirs(unit_path: Path, modes: list[str]) -> list[Path]:
    directories = [
        unit_path / "artifacts" / MODE_TO_ARTIFACT_DIR[mode]
        if MODE_TO_ARTIFACT_DIR[mode]
        else unit_path / "artifacts"
        for mode in modes
    ]
    return list(dict.fromkeys(directories))


def _artifact_snapshot(directories: list[Path]) -> dict[Path, tuple[int, int]]:
    snapshot: dict[Path, tuple[int, int]] = {}
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                stat = path.stat()
                snapshot[path] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _resolve_unit(unit_arg: str) -> tuple[Path, str]:
    requested = Path(unit_arg)
    unit_path = requested if requested.is_absolute() else ROOT / requested
    unit_path = unit_path.resolve()

    try:
        display = str(unit_path.relative_to(ROOT))
    except ValueError:
        display = str(unit_path)

    return unit_path, display


def _select_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError as exc:
        if requested == "cuda":
            raise SystemExit("--device cuda requires a PyTorch installation with CUDA support") from exc
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda was requested, but CUDA is unavailable. "
            "Use --device cpu or run in a CUDA-enabled environment."
        )
    return "cuda" if torch.cuda.is_available() else "cpu"


def _script_for_mode(metadata: dict[str, LessonValue], mode: str) -> str:
    scripts = metadata.get("scripts")
    if isinstance(scripts, dict) and mode in scripts:
        return scripts[mode]
    if isinstance(scripts, list):
        expected_stem = Path(MODE_TO_SCRIPT[mode]).stem
        for filename in scripts:
            if Path(filename).stem == expected_stem:
                return filename
    return MODE_TO_SCRIPT[mode]


def _run_entrypoint(target_path: Path) -> None:
    previous_argv = sys.argv
    sys.argv = [str(target_path)]
    sys.path.insert(0, str(target_path.parent))
    try:
        try:
            runpy.run_path(str(target_path), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
    finally:
        sys.path.remove(str(target_path.parent))
        sys.argv = previous_argv


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a lesson's scratch, framework, and analysis entrypoints.")
    parser.add_argument("--unit", required=True, help="Unit path, e.g. 00_foundations/01_tensor_shapes")
    parser.add_argument("--mode", choices=[*MODE_ORDER, "stage", "all"], required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Execution device. Defaults to CPU; use auto/cuda only after checking an idle GPU.",
    )
    args = parser.parse_args()

    unit_path, display_unit = _resolve_unit(args.unit)
    lesson_path = unit_path / "lesson.yaml"
    if not lesson_path.exists():
        raise SystemExit(f"lesson metadata not found: {lesson_path}")

    metadata = load_lesson_metadata(lesson_path)
    if args.mode == "all" and (unit_path / "run_stage.py").is_file() and not (unit_path / "scratch_lab.py").is_file():
        modes = ["stage"]
    else:
        modes = list(MODE_ORDER) if args.mode == "all" else [args.mode]
    targets = [(mode, unit_path / _script_for_mode(metadata, mode)) for mode in modes]
    missing = [(mode, path) for mode, path in targets if not path.is_file()]
    if missing:
        details = "\n".join(
            f"- --mode {mode} expects {path}"
            for mode, path in missing
        )
        raise SystemExit(
            "lesson entrypoint is missing. Add the declared script or choose an available mode:\n"
            f"{details}"
        )

    artifact_dirs = _artifact_dirs(unit_path, modes)
    before_artifacts = _artifact_snapshot(artifact_dirs)
    selected_device = _select_device(args.device)
    os.environ["BTB_DEVICE"] = selected_device
    if selected_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print(
        " ".join(
            [
                "run_context",
                f"unit={display_unit}",
                f"requested_mode={args.mode}",
                f"selected_device={selected_device}",
                f"objective={metadata.get('objective', '')}",
            ]
        ),
        flush=True,
    )

    for _mode, target_path in targets:
        _run_entrypoint(target_path)

    artifact_paths = {
        path
        for artifact_dir in artifact_dirs
        if artifact_dir.exists()
        for path in artifact_dir.rglob("*")
        if path.is_file()
        and before_artifacts.get(path) != (path.stat().st_mtime_ns, path.stat().st_size)
    }
    artifact_files = sorted(_display_path(path) for path in artifact_paths)
    print(
        " ".join(
            [
                f"unit={display_unit}",
                f"mode={args.mode}",
                f"completed_modes={','.join(modes)}",
                f"selected_device={selected_device}",
                f"objective={metadata.get('objective', '')}",
                "artifact_dirs=" + ",".join(_display_path(path) for path in artifact_dirs),
                "artifacts=" + (",".join(artifact_files) if artifact_files else "none-observed"),
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
