from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.brain.runtime.screen_quiz_parser import parse_screen_quiz_state


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(payload, dict):
        rows = payload.get("samples")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _resolve_path(base: Path, raw: Any) -> Optional[Path]:
    if not raw:
        return None
    p = Path(str(raw))
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _expected_control_kind(expected_global_type: str, expected_block_types: List[str]) -> str:
    gt = str(expected_global_type or "").strip().lower()
    blocks = [str(v or "").strip().lower() for v in expected_block_types if str(v or "").strip()]
    if gt in {"dropdown", "dropdown_scroll"}:
        return "dropdown"
    if gt == "text":
        return "text"
    if gt == "slider":
        return "slider"
    if gt in {"single", "multi"}:
        return "choice"
    if gt in {"triple", "mixed"}:
        if "slider" in blocks:
            return "slider"
        if any(b in {"dropdown", "dropdown_scroll"} for b in blocks):
            return "dropdown"
        if "text" in blocks and all(b == "text" for b in blocks):
            return "text"
        return "choice"
    return "unknown"


def _artifact_has_signal(region_payload: Dict[str, Any], summary_data: Optional[Dict[str, Any]], rated_data: Optional[Dict[str, Any]]) -> bool:
    region_results = region_payload.get("results") if isinstance(region_payload.get("results"), list) else []
    if region_results:
        return True
    if isinstance(summary_data, dict):
        if int(summary_data.get("total_elements") or 0) > 0:
            return True
        if summary_data.get("question_candidate") or summary_data.get("answer_candidate_boxes") or summary_data.get("dropdown_candidate_boxes") or summary_data.get("top_labels"):
            return True
    if isinstance(rated_data, dict):
        if int(rated_data.get("total_elements") or 0) > 0:
            return True
        summary = rated_data.get("summary") if isinstance(rated_data.get("summary"), dict) else {}
        if any(int(summary.get(k) or 0) > 0 for k in ("next_detected", "dropdown_detected", "question_detected", "cookies_detected")):
            return True
    return False


def _feature_row(sample: Dict[str, Any], screen_state: Dict[str, Any]) -> Dict[str, Any]:
    questions = screen_state.get("questions") if isinstance(screen_state.get("questions"), list) else []
    blocks = screen_state.get("blocks") if isinstance(screen_state.get("blocks"), list) else []
    active_block = screen_state.get("active_block") if isinstance(screen_state.get("active_block"), dict) else {}
    options = screen_state.get("options") if isinstance(screen_state.get("options"), list) else []
    features = screen_state.get("quiz_type_features") if isinstance(screen_state.get("quiz_type_features"), dict) else {}
    signals = screen_state.get("type_signals") if isinstance(screen_state.get("type_signals"), dict) else {}
    artifact_flags = signals.get("artifact_prompt_flags") if isinstance(signals.get("artifact_prompt_flags"), dict) else {}
    return {
        "id": str(sample.get("id") or ""),
        "source_view_id": str(sample.get("source_view_id") or sample.get("id") or ""),
        "expected_global_type": str(sample.get("expected_global_type") or ""),
        "expected_block_types": sample.get("expected_block_types") or [],
        "expected_control_kind": _expected_control_kind(
            str(sample.get("expected_global_type") or ""),
            list(sample.get("expected_block_types") or []),
        ),
        "parser_control_kind": str(screen_state.get("control_kind") or "unknown"),
        "parser_detected_quiz_type": str(screen_state.get("detected_quiz_type") or "unknown"),
        "active_block_type": str(screen_state.get("active_block_type") or ""),
        "question_text": str(screen_state.get("question_text") or ""),
        "question_count": len(questions),
        "block_count": len(blocks),
        "answer_count": len(options),
        "has_next": 1 if screen_state.get("next_bbox") else 0,
        "has_select": 1 if screen_state.get("select_bbox") else 0,
        "has_input": 1 if screen_state.get("input_bbox") else 0,
        "scroll_needed": 1 if screen_state.get("scroll_needed") else 0,
        "type_confidence": float(screen_state.get("type_confidence") or 0.0),
        "decision_margin": float(screen_state.get("decision_margin") or 0.0),
        "active_block_family": str(active_block.get("block_family") or ""),
        "active_block_control_kind": str(active_block.get("control_kind") or ""),
        "artifact_triple_hint": 1 if artifact_flags.get("triple_hint") else 0,
        "artifact_mix_hint": 1 if artifact_flags.get("mix_hint") else 0,
        "artifact_slider_hint": 1 if artifact_flags.get("slider_hint") else 0,
        "artifact_scroll_hint": 1 if artifact_flags.get("scroll_hint") else 0,
        "artifact_multi_hint": 1 if artifact_flags.get("multi_hint") else 0,
        "quiz_type_features": features,
        "type_signals": signals,
        "questions": questions,
        "blocks": blocks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build control_kind/question-grouping training dataset from screen benchmark.")
    ap.add_argument("--manifest", required=True, help="Screen benchmark manifest JSON.")
    ap.add_argument("--out-jsonl", required=True, help="Output JSONL path.")
    ap.add_argument("--max-samples", type=int, default=0, help="Optional sample cap.")
    ap.add_argument("--skip-empty-artifacts", type=int, default=1, help="Skip rows where region/summary/rated carry no usable signal.")
    ap.add_argument("--skip-empty-parser", type=int, default=1, help="Skip rows where parser returns no questions and no blocks.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f"[ERROR] manifest not found: {manifest_path}")
        return 2
    rows = _load_manifest(manifest_path)
    if args.max_samples > 0:
        rows = rows[: int(args.max_samples)]
    if not rows:
        print(f"[ERROR] no rows in manifest: {manifest_path}")
        return 2

    out_path = Path(args.out_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_dir = manifest_path.parent

    written = 0
    skipped = 0
    skipped_empty_artifacts = 0
    skipped_empty_parser = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            region_path = _resolve_path(manifest_dir, row.get("region_json"))
            if not isinstance(region_path, Path) or not region_path.exists():
                skipped += 1
                continue
            summary_path = _resolve_path(manifest_dir, row.get("summary_json"))
            rated_path = _resolve_path(manifest_dir, row.get("rated_json"))
            page_path = _resolve_path(manifest_dir, row.get("page_json"))
            region_payload = _load_json(region_path)
            if not isinstance(region_payload, dict):
                skipped += 1
                continue
            summary_data = _load_json(summary_path) if isinstance(summary_path, Path) and summary_path.exists() else None
            rated_data = _load_json(rated_path) if isinstance(rated_path, Path) and rated_path.exists() else None
            page_data = _load_json(page_path) if isinstance(page_path, Path) and page_path.exists() else None
            if int(args.skip_empty_artifacts or 0):
                if not _artifact_has_signal(region_payload, summary_data, rated_data):
                    skipped += 1
                    skipped_empty_artifacts += 1
                    continue
            screen_state = parse_screen_quiz_state(
                region_payload=region_payload,
                summary_data=summary_data,
                rated_data=rated_data,
                page_data=page_data,
            )
            if int(args.skip_empty_parser or 0):
                questions = screen_state.get("questions") if isinstance(screen_state.get("questions"), list) else []
                blocks = screen_state.get("blocks") if isinstance(screen_state.get("blocks"), list) else []
                if not questions and not blocks:
                    skipped += 1
                    skipped_empty_parser += 1
                    continue
            feature_row = _feature_row(row, screen_state)
            f.write(json.dumps(feature_row, ensure_ascii=False) + "\n")
            written += 1

    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "out_jsonl": str(out_path),
                "written": written,
                "skipped": skipped,
                "skipped_empty_artifacts": skipped_empty_artifacts,
                "skipped_empty_parser": skipped_empty_parser,
            },
            ensure_ascii=False,
        )
    )
    return 0 if written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
