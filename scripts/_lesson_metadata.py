from __future__ import annotations

from pathlib import Path


LessonValue = str | list[str]


def _parse_scalar(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_lesson_metadata(path: str | Path) -> dict[str, LessonValue]:
    """Parse the repo's constrained lesson.yaml format without extra deps.

    Supported schema:
    - top-level `key: value` string pairs
    - top-level `key:` followed by `  - item` lists
    - blank lines and comment lines beginning with `#`
    """

    metadata: dict[str, LessonValue] = {}
    current_key: str | None = None
    path = Path(path)

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if raw_line.startswith("  - "):
            if current_key is None:
                raise ValueError(f"{path}:{line_number}: list item without a preceding key")
            current_value = metadata.get(current_key)
            if not isinstance(current_value, list):
                raise ValueError(f"{path}:{line_number}: key '{current_key}' does not accept list items")
            current_value.append(raw_line[4:].strip())
            continue

        if raw_line.startswith(" "):
            raise ValueError(f"{path}:{line_number}: unsupported indentation in constrained lesson.yaml")

        key, separator, value = raw_line.partition(":")
        if separator != ":":
            raise ValueError(f"{path}:{line_number}: expected 'key: value' format")

        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{line_number}: empty key is not allowed")

        current_key = key
        value = value.strip()
        metadata[current_key] = [] if value == "" else _parse_scalar(value)

    return metadata

