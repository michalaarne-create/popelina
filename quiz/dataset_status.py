from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def _count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob(pattern))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Show dataset generation progress, including shard state.")
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--target-count", type=int, default=0, help="Optional target sample count for percentage/ETA.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    parts_root = out_dir / "_parts"
    images = _count_files(out_dir / "images", "*.png")
    labels = _count_files(out_dir / "labels", "*.json")

    parts: List[Dict[str, Any]] = []
    if parts_root.exists():
        for part_dir in sorted([p for p in parts_root.iterdir() if p.is_dir()]):
            summary_path = part_dir / "manifests" / "dataset_summary.json"
            part_images = _count_files(part_dir / "images", "*.png")
            part_labels = _count_files(part_dir / "labels", "*.json")
            row: Dict[str, Any] = {
                "part": part_dir.name,
                "images": int(part_images),
                "labels": int(part_labels),
                "complete": False,
            }
            if summary_path.exists():
                try:
                    summary = _load_json(summary_path)
                    row["count_requested"] = int(summary.get("count_requested") or 0)
                    row["total_samples"] = int(summary.get("total_samples") or 0)
                    row["total_views"] = int(summary.get("total_views") or 0)
                    row["complete"] = not bool(summary.get("errors"))
                except Exception:
                    row["summary_error"] = True
            parts.append(row)

    shard_images = sum(int(p.get("images") or 0) for p in parts)
    shard_labels = sum(int(p.get("labels") or 0) for p in parts)
    effective_images = max(images, shard_images)
    effective_labels = max(labels, shard_labels)
    target = int(args.target_count)
    progress = 0.0
    if target > 0:
        progress = min(1.0, float(effective_images) / float(target))

    out = {
        "out_dir": str(out_dir),
        "images": int(images),
        "labels": int(labels),
        "shard_images": int(shard_images),
        "shard_labels": int(shard_labels),
        "effective_images": int(effective_images),
        "effective_labels": int(effective_labels),
        "target_count": int(target),
        "progress_ratio": float(progress),
        "progress_percent": float(round(progress * 100.0, 2)) if target > 0 else 0.0,
        "remaining_estimate": int(max(0, target - effective_images)) if target > 0 else None,
        "parts": parts,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
