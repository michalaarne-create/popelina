from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_quiz_type_model import _read_manifest, _resolve
from scripts.brain.runtime.quiz_type_classifier import classify_quiz_type


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _question_from_block(block: Dict[str, Any], idx: int, has_next: bool, require_scroll: bool) -> Dict[str, Any]:
    block_type = str(block.get("type") or "single")
    control_kind = "choice"
    select_bbox = None
    input_bbox = None
    if block_type in {"dropdown", "dropdown_scroll"}:
        control_kind = "dropdown"
        select_bbox = [20, 100 + (idx * 120), 920, 150 + (idx * 120)]
    elif block_type == "text":
        control_kind = "text"
        input_bbox = [20, 100 + (idx * 120), 920, 150 + (idx * 120)]
    elif block_type == "slider":
        control_kind = "slider"
    options = []
    for opt_idx, text in enumerate(block.get("options") or []):
        options.append(
            {
                "text": str(text),
                "norm_text": str(text).strip().lower(),
                "bbox": [40, 160 + (idx * 120) + (opt_idx * 20), 820, 178 + (idx * 120) + (opt_idx * 20)],
            }
        )
    return {
        "id": str(block.get("block_id") or f"b{idx+1}"),
        "prompt_text": str(block.get("prompt") or ""),
        "question_text": str(block.get("prompt") or ""),
        "control_kind": control_kind,
        "block_type": block_type,
        "options": options,
        "select_bbox": select_bbox,
        "input_bbox": input_bbox,
        "scroll_needed": bool(require_scroll or block_type == "dropdown_scroll"),
        "next_bbox": [840, 1780, 1040, 1840] if has_next else None,
        "bbox": [20, 40 + (idx * 120), 980, 90 + (idx * 120)],
        "confidence": 1.0,
    }


def _marker_bbox(option_bbox: List[int]) -> List[int]:
    top = int(option_bbox[1])
    bottom = int(option_bbox[3])
    center_y = (top + bottom) // 2
    return [12, center_y - 8, 28, center_y + 8]


def _rated_data_from_questions(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    elements: List[Dict[str, Any]] = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        block_type = str(question.get("block_type") or "")
        control_kind = str(question.get("control_kind") or "")
        effective_type = block_type or ("dropdown" if control_kind == "dropdown" else "text" if control_kind == "text" else "single")
        marker_kind = ""
        marker_shape = ""
        if effective_type == "single":
            marker_kind = "radio"
            marker_shape = "circle"
        elif effective_type == "multi":
            marker_kind = "checkbox"
            marker_shape = "square"
        if not marker_kind:
            continue
        options = question.get("options") if isinstance(question.get("options"), list) else []
        for opt in options:
            if not isinstance(opt, dict):
                continue
            bbox = opt.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            elements.append(
                {
                    "text": str(opt.get("text") or ""),
                    "bbox": list(bbox),
                    "marker": {
                        "kind": marker_kind,
                        "shape": marker_shape,
                        "conf": 0.98,
                        "bbox": _marker_bbox(list(bbox)),
                    },
                }
            )
    return {"elements": elements, "total_elements": len(elements)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark runtime quiz_type classifier from label manifest.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--disable-guardrails", action="store_true")
    args = parser.parse_args()
    if args.disable_guardrails:
        os.environ["FULLBOT_QUIZ_TYPE_GUARDRAILS"] = "0"

    manifest_path = Path(args.manifest).resolve()
    rows = _read_manifest(manifest_path)
    confusion: Dict[str, Dict[str, int]] = defaultdict(dict)
    per_expected: Dict[str, Counter[str]] = defaultdict(Counter)
    results: List[Dict[str, Any]] = []

    for row in rows:
        expected = str(row.get("expected_global_type") or "").strip()
        label_path = _resolve(str(row.get("label_path") or ""), manifest_path)
        label = _load_json(label_path)
        blocks = label.get("blocks") if isinstance(label.get("blocks"), list) else []
        questions = [
            _question_from_block(
                block=block,
                idx=idx,
                has_next=bool(label.get("has_next")),
                require_scroll=bool(label.get("require_scroll")),
            )
            for idx, block in enumerate(blocks)
            if isinstance(block, dict)
        ]
        active = questions[0] if questions else None
        rated_data = _rated_data_from_questions(questions)
        out = classify_quiz_type(
            region_payload={},
            summary_data=None,
            rated_data=rated_data,
            questions=questions,
            active=active,
            screen_w=int((label.get("viewport") or {}).get("width") or 1080),
            screen_h=int((label.get("viewport") or {}).get("height") or 1920),
        )
        predicted = str(out.get("detected_quiz_type") or "unknown")
        per_expected[expected][predicted] += 1
        results.append(
            {
                "sample_id": str(row.get("sample_id") or ""),
                "expected": expected,
                "predicted": predicted,
                "confidence": float(out.get("type_confidence") or 0.0),
                "reason": str(out.get("type_reason") or ""),
            }
        )

    total = len(results)
    correct = sum(1 for row in results if row["expected"] == row["predicted"])
    per_class: Dict[str, Dict[str, float]] = {}
    for expected, counts in per_expected.items():
        n = sum(counts.values())
        correct_n = int(counts.get(expected, 0))
        confusion[expected] = {pred: int(count) for pred, count in counts.items()}
        per_class[expected] = {"n": n, "acc": (float(correct_n) / float(n)) if n else 0.0}

    report = {
        "rows": total,
        "accuracy": (float(correct) / float(total)) if total else 0.0,
        "per_class": per_class,
        "confusion": confusion,
    }
    if args.out_json:
        out_path = Path(args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
