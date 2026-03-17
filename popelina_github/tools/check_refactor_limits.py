from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _count_lines(path: Path) -> int:
    return sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate max line count for refactor scope.")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--max-lines", type=int, default=200)
    ap.add_argument("--exclude-main", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    targets = [
        *sorted((root / "scripts" / "pipeline").glob("*.py")),
        *sorted((root / "scripts" / "debuggers").glob("*.py")),
        *sorted((root / "scripts" / "brain").glob("*.py")),
    ]
    if not args.exclude_main:
        targets.append(root / "main.py")

    failed = []
    for path in sorted(set(targets)):
        if not path.exists():
            continue
        lines = _count_lines(path)
        rel = path.relative_to(root)
        print(f"{lines:5d} {rel}")
        if lines > int(args.max_lines):
            failed.append((lines, str(rel)))

    if failed:
        print("\nFAIL: line limit exceeded:")
        for lines, rel in failed:
            print(f" - {rel}: {lines} > {args.max_lines}")
        return 1
    print("\nOK: all files within line limit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
