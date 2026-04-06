from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from random_quiz_sandbox import GLOBAL_TYPES, build_sample


SUPPORTED_DEFAULT = ["single", "multi", "dropdown", "dropdown_scroll", "slider", "text", "triple", "mixed"]


def _parse_allowed(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return list(SUPPORTED_DEFAULT)
    out: List[str] = []
    for part in text.split(","):
        name = str(part or "").strip()
        if not name:
            continue
        if name not in GLOBAL_TYPES:
            raise ValueError(f"Unsupported global type: {name}")
        out.append(name)
    if not out:
        raise ValueError("No allowed global types after parsing.")
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build label-only dataset for quiz_type training/eval.")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260319)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--difficulty", type=str, default="hard")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--allowed-global-types", type=str, default=",".join(SUPPORTED_DEFAULT))
    parser.add_argument("--balanced", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    labels_dir = out_dir / "labels"
    manifests_dir = out_dir / "manifests"
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    allowed = _parse_allowed(args.allowed_global_types)
    count = max(1, int(args.count))
    start_index = int(args.start_index)
    difficulty = str(args.difficulty or "hard")
    seed = int(args.seed)
    balanced = bool(int(args.balanced))

    forced_types: List[str] = []
    if balanced:
        for i in range(count):
            forced_types.append(allowed[(start_index + i) % len(allowed)])
        rng = random.Random(seed * 17 + 91 + start_index)
        rng.shuffle(forced_types)

    rows: List[Dict[str, Any]] = []
    for i in range(count):
        sample = build_sample(
            seed=seed,
            index=start_index + i,
            forced_global_type=(forced_types[i] if forced_types else None),
            difficulty=difficulty,
        )
        sample_id = str(sample["sample_id"])
        label_path = labels_dir / f"{sample_id}.json"
        label_payload = {
            "sample_id": sample_id,
            "global_type": sample["global_type"],
            "block_types": sample["block_types"],
            "profile": sample.get("profile") or difficulty,
            "has_next": sample["has_next"],
            "auto_next": sample["auto_next"],
            "require_scroll": sample["require_scroll"],
            "partial_next_question_visible": bool(sample.get("partial_next_question_visible")),
            "blocks": sample["blocks"],
            "style": sample["style"],
            "viewport": sample["viewport"],
        }
        _write_json(label_path, label_payload)
        rows.append(
            {
                "sample_id": sample_id,
                "view_id": f"{sample_id}__label",
                "expected_global_type": sample["global_type"],
                "expected_block_types": sample["block_types"],
                "label_path": str(label_path.relative_to(out_dir)).replace("\\", "/"),
            }
        )

    manifest_json = manifests_dir / "dataset_manifest.json"
    manifest_jsonl = manifests_dir / "dataset_manifest.jsonl"
    _write_json(manifest_json, {"rows": rows})
    manifest_jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "count": count,
                "allowed_global_types": allowed,
                "balanced": balanced,
                "manifest_json": str(manifest_json),
                "manifest_jsonl": str(manifest_jsonl),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
