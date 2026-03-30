from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INLINE_LINK_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)|\[[^\]]+\]\(([^)]+)\)")
SKIP_SCHEMES = ("http://", "https://", "mailto:", "tel:", "data:", "javascript:")
SCAN_ROOTS = [
    ROOT / "README.md",
    ROOT / "docs",
    ROOT / "00_foundations",
    ROOT / "00_shared",
    ROOT / "01_ml",
    ROOT / "02_nlp_bridge",
    ROOT / "03_nlp",
    ROOT / "04_multimodal_bridge",
    ROOT / "05_multimodal",
]


def iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if root.is_file():
            files.append(root)
        elif root.exists():
            files.extend(sorted(root.rglob("*.md")))
    return files


def _iter_links(markdown_text: str) -> list[str]:
    links: list[str] = []
    in_fenced_block = False

    for raw_line in markdown_text.splitlines():
        if raw_line.lstrip().startswith("```"):
            in_fenced_block = not in_fenced_block
            continue
        if in_fenced_block:
            continue
        for image_link, text_link in INLINE_LINK_RE.findall(raw_line):
            link = image_link or text_link
            links.append(link.strip())

    return links


def _is_local_target(link: str) -> bool:
    return bool(link) and not link.startswith("#") and not link.startswith(SKIP_SCHEMES)


def _normalize_target(markdown_file: Path, link: str) -> Path | None:
    target = link.split("#", maxsplit=1)[0].strip()
    if not target:
        return None
    return (markdown_file.parent / target).resolve()


def main() -> int:
    missing: list[str] = []

    for markdown_file in iter_markdown_files():
        text = markdown_file.read_text(encoding="utf-8")
        for link in _iter_links(text):
            if not _is_local_target(link):
                continue

            target = _normalize_target(markdown_file, link)
            if target is None or target.exists():
                continue

            missing.append(f"{markdown_file.relative_to(ROOT)} -> {link}")

    if missing:
        print("Missing local markdown links:", file=sys.stderr)
        for entry in sorted(missing):
            print(entry, file=sys.stderr)
        return 1

    print("OK: curriculum markdown links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
