from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
AUTO_RUNS_DIR = ROOT / "data" / "auto_main" / "runs"
DEFAULT_QA_CACHE = ROOT / "data" / "answers" / "qa_cache.json"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_name(value: str) -> str:
    raw = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").strip())
    collapsed = "_".join(part for part in raw.split("_") if part)
    return collapsed or "item"


def _run_root_from_input(path: Path) -> Path:
    current = path.resolve()
    if current.is_file():
        current = current.parent
    if current.name.startswith("iter_"):
        return current.parent
    return current


def _iter_dirs(run_dir: Path) -> List[Path]:
    return sorted([path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("iter_")])


def _resolve_run_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return _run_root_from_input(candidate)
    fallback = AUTO_RUNS_DIR / str(path)
    if fallback.exists():
        return _run_root_from_input(fallback)
    raise FileNotFoundError(f"Run directory not found: {path}")


def _load_qa_cache(cache_path: Path = DEFAULT_QA_CACHE) -> Dict[str, Any]:
    payload = _load_json(cache_path)
    items = payload.get("items")
    if not isinstance(items, dict):
        raise RuntimeError(f"Invalid QA cache payload: {cache_path}")
    return items


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).lower()


@dataclass
class IterationReplayRecord:
    iteration_dir: Path
    question_text: str
    stored_plan_kind: str
    replay_plan_kind: str
    stored_reason: str
    replay_reason: str
    action_count: int
    replay_action_count: int
    matched: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration_dir": str(self.iteration_dir),
            "question_text": self.question_text,
            "stored_plan_kind": self.stored_plan_kind,
            "replay_plan_kind": self.replay_plan_kind,
            "stored_reason": self.stored_reason,
            "replay_reason": self.replay_reason,
            "action_count": self.action_count,
            "replay_action_count": self.replay_action_count,
            "matched": self.matched,
        }


def _cache_item_for_screen_state(screen_state: Dict[str, Any], qa_cache: Dict[str, Any]) -> Dict[str, Any]:
    question_norm = _normalize_text(screen_state.get("question_text") or "")
    option_norms = sorted(
        _normalize_text((row or {}).get("text") or "")
        for row in (screen_state.get("options") or [])
        if isinstance(row, dict)
    )
    for item in qa_cache.values():
        if not isinstance(item, dict):
            continue
        if _normalize_text(item.get("question_text") or "") != question_norm:
            continue
        item_opts = sorted(_normalize_text(value) for value in (item.get("options_text") or {}).values())
        if option_norms and item_opts and option_norms != item_opts:
            continue
        return item
    return {}


def _build_dom_candidate_actions(screen_state: Dict[str, Any], cache_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    options = screen_state.get("options") if isinstance(screen_state.get("options"), list) else []
    next_bbox = screen_state.get("next_bbox") if isinstance(screen_state.get("next_bbox"), list) else None
    selected_labels: List[str] = []
    options_text = cache_item.get("options_text") if isinstance(cache_item.get("options_text"), dict) else {}
    for key in cache_item.get("selected_options") or []:
        value = options_text.get(key)
        if value:
            selected_labels.append(str(value))
    if not selected_labels and cache_item.get("correct_answer"):
        selected_labels.append(str(cache_item.get("correct_answer")))

    actions: List[Dict[str, Any]] = []
    for label in selected_labels:
        for row in options:
            if not isinstance(row, dict):
                continue
            if _normalize_text(row.get("text") or "") != _normalize_text(label):
                continue
            bbox = row.get("bbox") if isinstance(row.get("bbox"), list) and len(row.get("bbox")) == 4 else None
            if bbox is not None:
                actions.append({"kind": "screen_click", "reason": f"click_answer:{label}", "bbox": bbox})
            break
    if actions and next_bbox and len(next_bbox) == 4:
        actions.append({"kind": "wait", "amount": 120, "reason": "wait_before_next"})
        actions.append({"kind": "screen_click", "reason": "click_next_after_answer", "bbox": next_bbox})
    return actions


def replay_artifact_run(run_dir: str | Path, cache_path: Path = DEFAULT_QA_CACHE) -> Dict[str, Any]:
    from scripts.brain.runtime.decision_core import build_decision_core

    resolved_run_dir = _resolve_run_dir(run_dir)
    qa_cache = _load_qa_cache(cache_path)
    records: List[IterationReplayRecord] = []
    for iter_dir in _iter_dirs(resolved_run_dir):
        decision_path = iter_dir / "decision.json"
        screen_state_path = iter_dir / "screen_state.json"
        if not decision_path.exists() or not screen_state_path.exists():
            continue
        stored_decision = _load_json(decision_path)
        screen_state = _load_json(screen_state_path)
        replay_core = build_decision_core(
            cache_path=cache_path,
            screen_state=screen_state,
            prev_state={},
            controls_data=None,
            page_data=None,
        )
        stored_actions = stored_decision.get("actions") if isinstance(stored_decision.get("actions"), list) else []
        replay_actions = replay_core.actions if isinstance(replay_core.actions, list) else []
        record = IterationReplayRecord(
            iteration_dir=iter_dir,
            question_text=str(screen_state.get("question_text") or ""),
            stored_plan_kind=str((((stored_actions or [{}])[0]) or {}).get("kind") or ""),
            replay_plan_kind=str((((replay_actions or [{}])[0]) or {}).get("kind") or ""),
            stored_reason=str((((stored_actions or [{}])[0]) or {}).get("reason") or ""),
            replay_reason=str((((replay_actions or [{}])[0]) or {}).get("reason") or ""),
            action_count=len(stored_actions),
            replay_action_count=len(replay_actions),
            matched=stored_actions == replay_actions,
        )
        records.append(record)
    return {
        "run_dir": str(resolved_run_dir),
        "iterations_total": len(records),
        "iterations_matched": sum(1 for item in records if item.matched),
        "records": [item.to_dict() for item in records],
    }


def simulate_counterfactuals(
    screen_state: Dict[str, Any],
    trace: Optional[Dict[str, Any]],
    candidates: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    expectation = (trace or {}).get("post_action_expectation") if isinstance((trace or {}).get("post_action_expectation"), dict) else {}
    expected_kind = str(expectation.get("expected_first_action_kind") or "")
    expected_values = [_normalize_text(value) for value in expectation.get("expected_values") or [] if _normalize_text(value)]
    option_texts = {_normalize_text((row or {}).get("text") or "") for row in (screen_state.get("options") or []) if isinstance(row, dict)}
    ranked: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        actions = candidate.get("actions") if isinstance(candidate.get("actions"), list) else []
        first = ((actions or [{}])[0]) or {}
        first_kind = str(first.get("kind") or "")
        first_reason = str(first.get("reason") or "")
        score = 0.0
        if expected_kind and first_kind == expected_kind:
            score += 0.55
        if "next" in first_reason and screen_state.get("next_bbox"):
            score += 0.2
        if "answer:" in first_reason:
            answer_label = _normalize_text(first_reason.split(":", 1)[1])
            if answer_label in option_texts:
                score += 0.2
            if expected_values and answer_label in expected_values:
                score += 0.15
        if first_kind == "dom_select_option" and screen_state.get("select_bbox"):
            score += 0.2
        if first_kind == "type_text" and screen_state.get("input_bbox"):
            score += 0.2
        ranked.append(
            {
                "rank_seed": idx,
                "candidate_name": str(candidate.get("name") or f"candidate_{idx}"),
                "first_kind": first_kind,
                "first_reason": first_reason,
                "action_count": len(actions),
                "score": round(score, 4),
            }
        )
    ranked.sort(key=lambda item: (-float(item["score"]), int(item["rank_seed"])))
    return {
        "expected_first_action_kind": expected_kind,
        "expected_values": expected_values,
        "ranked_candidates": ranked,
    }


def diff_decisions(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    left_actions = left.get("actions") if isinstance(left.get("actions"), list) else []
    right_actions = right.get("actions") if isinstance(right.get("actions"), list) else []
    left_resolved = left.get("resolved_answer") if isinstance(left.get("resolved_answer"), dict) else {}
    right_resolved = right.get("resolved_answer") if isinstance(right.get("resolved_answer"), dict) else {}
    left_trace = left.get("trace") if isinstance(left.get("trace"), dict) else {}
    right_trace = right.get("trace") if isinstance(right.get("trace"), dict) else {}
    return {
        "question_left": str((left.get("screen_state") or {}).get("question_text") or ""),
        "question_right": str((right.get("screen_state") or {}).get("question_text") or ""),
        "action_kind_left": str((((left_actions or [{}])[0]) or {}).get("kind") or ""),
        "action_kind_right": str((((right_actions or [{}])[0]) or {}).get("kind") or ""),
        "action_reason_left": str((((left_actions or [{}])[0]) or {}).get("reason") or ""),
        "action_reason_right": str((((right_actions or [{}])[0]) or {}).get("reason") or ""),
        "resolved_answers_left": [str(v) for v in (left_resolved.get("correct_answers") or [])],
        "resolved_answers_right": [str(v) for v in (right_resolved.get("correct_answers") or [])],
        "stage_left": str(left_trace.get("stage") or ""),
        "stage_right": str(right_trace.get("stage") or ""),
        "actions_equal": left_actions == right_actions,
        "resolved_equal": left_resolved == right_resolved,
        "trace_equal": left_trace == right_trace,
    }


def build_session_bundle(run_dir: str | Path, output_root: Path) -> Dict[str, Any]:
    resolved_run_dir = _resolve_run_dir(run_dir)
    summary_path = resolved_run_dir / "summary.json"
    trace_path = resolved_run_dir / "trace.jsonl"
    summary = _load_json(summary_path) if summary_path.exists() else {}
    bundle_name = f"{resolved_run_dir.name}_bundle"
    bundle_dir = output_root / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    copied_files: List[str] = []
    for source in [summary_path, trace_path]:
        if source.exists():
            target = bundle_dir / source.name
            shutil.copy2(source, target)
            copied_files.append(str(target))

    iter_manifest: List[Dict[str, Any]] = []
    for iter_dir in _iter_dirs(resolved_run_dir):
        target_iter_dir = bundle_dir / iter_dir.name
        target_iter_dir.mkdir(parents=True, exist_ok=True)
        copied_iter_files: List[str] = []
        for name in ("screen_state.json", "decision.json", "state.json", "page.html"):
            source = iter_dir / name
            if not source.exists():
                continue
            target = target_iter_dir / name
            shutil.copy2(source, target)
            copied_iter_files.append(str(target))
        iter_manifest.append({"iteration_dir": iter_dir.name, "files": copied_iter_files})

    manifest = {
        "bundle_dir": str(bundle_dir),
        "source_run_dir": str(resolved_run_dir),
        "summary_completed": bool(summary.get("completed")),
        "iterations": len(iter_manifest),
        "copied_root_files": copied_files,
        "copied_iterations": iter_manifest,
        "reproduce_command": f"python \"{ROOT / 'Replay and simulation' / 'artifact_replay_runner.py'}\" --run-dir \"{bundle_dir}\"",
    }
    _dump_json(bundle_dir / "manifest.json", manifest)
    return manifest


def build_counterfactual_candidates(screen_state: Dict[str, Any], decision: Dict[str, Any], qa_cache: Dict[str, Any]) -> List[Dict[str, Any]]:
    stored_actions = decision.get("actions") if isinstance(decision.get("actions"), list) else []
    cache_item = _cache_item_for_screen_state(screen_state, qa_cache)
    dom_actions = _build_dom_candidate_actions(screen_state, cache_item) if cache_item else []
    candidates = [
        {"name": "stored_decision", "actions": stored_actions},
        {"name": "dom_counterfactual", "actions": dom_actions},
        {"name": "noop_guard", "actions": [{"kind": "noop", "reason": "manual_noop"}]},
    ]
    return candidates
