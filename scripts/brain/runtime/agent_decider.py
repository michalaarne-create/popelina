from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .action_planner import plan_actions
from .agent_state_io import load_json, load_state, save_state
from .agent_targets import select_target
from .answer_resolver import resolve_answer
from .pipeline_state_builder import build_brain_state
from .quiz_state_builder import build_quiz_state
from .readback_verifier import evaluate_transition


def _default_logger(message: str) -> None:
    print(message)


def _first_action_to_legacy(actions: List[Dict[str, Any]]) -> str:
    if not actions:
        return "idle"
    first = actions[0] if isinstance(actions[0], dict) else {}
    kind = str(first.get("kind") or "")
    reason = str(first.get("reason") or "")
    if kind == "screen_click":
        return "click_next" if "next" in reason else "click_answer"
    if kind == "screen_scroll":
        return "scroll_page_down"
    return "idle"


@dataclass
class BrainDecision:
    recommended_action: str
    target_bbox: Optional[List[float]]
    target_element: Optional[Dict[str, Any]]
    requires_action: bool
    brain_state: dict
    question_data: Optional[dict]
    summary_data: Optional[dict]
    actions: List[Dict[str, Any]] = field(default_factory=list)
    screen_state: Optional[Dict[str, Any]] = None
    answer_source: Optional[str] = None
    fallback_used: bool = False
    page_signature: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None


class PipelineBrainAgent:
    def __init__(
        self,
        question_path: Path,
        state_path: Path,
        logger: Optional[Callable[[str], None]] = None,
        *,
        controls_path: Optional[Path] = None,
        page_path: Optional[Path] = None,
        quiz_answer_cache: Optional[Path] = None,
        current_run_dir: Optional[Path] = None,
    ):
        self.question_path = question_path
        self.state_path = state_path
        self.controls_path = controls_path
        self.page_path = page_path
        self.quiz_answer_cache = quiz_answer_cache or (Path(__file__).resolve().parents[3] / "quiz" / "data" / "qa_cache.json")
        self.current_run_dir = current_run_dir
        self.logger = logger or _default_logger

    def decide(
        self,
        summary_path: Path,
        *,
        region_json_path: Optional[Path] = None,
        screenshot_path: Optional[Path] = None,
    ) -> BrainDecision:
        if self._quiz_mode_enabled():
            return self._decide_quiz(
                summary_path,
                region_json_path=region_json_path,
                screenshot_path=screenshot_path,
            )
        return self._decide_legacy(summary_path)

    def load_state(self) -> dict:
        return load_state(self.state_path, self._log)

    def _decide_legacy(self, summary_path: Path) -> BrainDecision:
        summary_data = load_json(summary_path, self._log)
        question_data = load_json(self.question_path, self._log)
        prev_state = load_state(self.state_path, self._log)
        if summary_data is None:
            self._error(f"Summary missing at {summary_path}; switching to idle.")
            return BrainDecision("idle", None, None, False, prev_state, question_data, None)

        consistency = self._screen_site_consistency()
        if consistency is not None and consistency < 0.5:
            state_dump = dict(prev_state or {})
            state_dump.update({"screen_site_consistency": float(consistency), "screen_site_matched": False})
            save_state(self.state_path, state_dump, self._log)
            return BrainDecision("idle", None, None, False, state_dump, question_data, summary_data)

        try:
            brain = build_brain_state(question_data, summary_data, prev_state, False, False)
        except Exception as exc:
            self._error(f"build_brain_state failed: {exc}; switching to idle.")
            return BrainDecision("idle", None, None, False, prev_state, question_data, summary_data)
        if not isinstance(brain, dict):
            self._error("build_brain_state returned non-dict; switching to idle.")
            return BrainDecision("idle", None, None, False, prev_state, question_data, summary_data)

        action = brain.get("recommended_action", "idle")
        target_element, target_bbox = select_target(brain, action)
        mark_answer = action == "click_answer" and target_bbox is not None
        mark_next = action == "click_next" and target_bbox is not None
        updated = build_brain_state(question_data, summary_data, prev_state, mark_answer, mark_next)
        save_state(self.state_path, updated, self._log)
        requires_action = action in {"click_answer", "click_next", "click_cookies_accept", "scroll_page_down"}
        return BrainDecision(action, target_bbox, target_element, requires_action, updated, question_data, summary_data)

    def _decide_quiz(
        self,
        summary_path: Path,
        *,
        region_json_path: Optional[Path],
        screenshot_path: Optional[Path],
    ) -> BrainDecision:
        prev_state = load_state(self.state_path, self._log)
        bundle = build_quiz_state(
            summary_path=summary_path,
            region_json_path=region_json_path,
            screenshot_path=screenshot_path,
            question_path=self.question_path,
            controls_path=self.controls_path,
            page_path=self.page_path,
        )
        screen_state = bundle.get("screen_state") or {}
        controls_data = bundle.get("controls_data")
        page_data = bundle.get("page_data")
        summary_data = bundle.get("summary_data")
        question_data = bundle.get("question_data")

        transition = evaluate_transition(
            prev_state=prev_state,
            current_screen_state=screen_state,
            controls_data=controls_data,
            page_data=page_data,
        )
        resolved = resolve_answer(
            cache_path=self._quiz_answer_cache(),
            screen_state=screen_state,
            controls_data=controls_data,
        )
        actions_objs, trace, fallback_used = plan_actions(
            screen_state=screen_state,
            resolved_answer=resolved,
            brain_state=prev_state,
            controls_data=controls_data,
            transition=transition,
        )
        actions = [action.to_dict() for action in actions_objs]
        legacy_action = _first_action_to_legacy(actions)
        target_element = None
        target_bbox = None
        if actions:
            first = actions[0]
            bbox = first.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                target_bbox = [float(v) for v in bbox]
                target_element = {
                    "kind": first.get("kind"),
                    "reason": first.get("reason"),
                    "bbox": target_bbox,
                }
        requires_action = bool(actions) and legacy_action != "idle"

        updated = self._build_quiz_brain_state(
            prev_state=prev_state,
            screen_state=screen_state,
            question_data=question_data,
            summary_data=summary_data,
            controls_data=controls_data,
            resolved_answer=resolved.to_dict(),
            actions=actions,
            legacy_action=legacy_action,
            transition=transition,
            fallback_used=bool(fallback_used),
            page_data=page_data,
            target_bbox=target_bbox,
        )
        self._write_quiz_debug(screen_state=screen_state, trace=trace, resolved=resolved.to_dict())
        save_state(self.state_path, updated, self._log)

        return BrainDecision(
            legacy_action,
            target_bbox,
            target_element,
            requires_action,
            updated,
            question_data,
            summary_data,
            actions=actions,
            screen_state=screen_state,
            answer_source=resolved.source,
            fallback_used=bool(fallback_used),
            page_signature=str(screen_state.get("page_signature") or ""),
            trace={
                **trace,
                "transition": transition,
                "resolved": resolved.to_dict(),
            },
        )

    def _build_quiz_brain_state(
        self,
        *,
        prev_state: Dict[str, Any],
        screen_state: Dict[str, Any],
        question_data: Optional[Dict[str, Any]],
        summary_data: Optional[Dict[str, Any]],
        controls_data: Optional[Dict[str, Any]],
        resolved_answer: Dict[str, Any],
        actions: List[Dict[str, Any]],
        legacy_action: str,
        transition: Dict[str, Any],
        fallback_used: bool,
        page_data: Optional[Dict[str, Any]],
        target_bbox: Optional[List[float]],
    ) -> Dict[str, Any]:
        attempts = dict((prev_state or {}).get("attempt_counts") or {})
        last_action = {}
        if actions:
            first = actions[0]
            expected_values = list(resolved_answer.get("correct_answers") or [])
            last_action = {
                "kind": "next" if legacy_action == "click_next" else ("answer" if legacy_action == "click_answer" else first.get("kind")),
                "raw_kind": first.get("kind"),
                "reason": first.get("reason"),
                "question_signature": screen_state.get("active_question_signature"),
                "page_signature": screen_state.get("page_signature") or (page_data or {}).get("page_signature"),
                "expected_values": expected_values,
                "bbox": target_bbox,
                "timestamp": time.time(),
            }
            attempt_key = "|".join(
                [
                    str(screen_state.get("screen_signature") or ""),
                    str(screen_state.get("active_question_signature") or ""),
                    str(first.get("kind") or ""),
                    ",".join(expected_values),
                    str(first.get("combo") or ""),
                ]
            )
            attempts[attempt_key] = int(attempts.get(attempt_key) or 0) + 1
        return {
            "timestamp": time.time(),
            "quiz_mode": True,
            "question_text": str(screen_state.get("question_text") or ""),
            "question_hash": str(screen_state.get("active_question_signature") or ""),
            "question_changed": bool(transition.get("question_changed")),
            "answer_clicked": legacy_action == "click_answer",
            "next_clicked": legacy_action == "click_next",
            "has_answers": bool(screen_state.get("options") or resolved_answer.get("question_type") in {"text", "dropdown", "dropdown_scroll"}),
            "has_next": bool(screen_state.get("next_bbox")),
            "recommended_action": legacy_action,
            "sources": {
                "question_json": str(self.question_path),
                "summary_json": str(summary_data.get("_source")) if isinstance(summary_data, dict) and summary_data.get("_source") else None,
                "controls_json": str(self.controls_path) if self.controls_path else None,
                "page_json": str(self.page_path) if self.page_path else None,
            },
            "objects": {
                "question": {
                    "text": screen_state.get("question_text"),
                    "bbox": ((screen_state.get("questions") or [{}])[0].get("bbox") if screen_state.get("questions") else None),
                },
                "answers": screen_state.get("options") or [],
                "next": {"bbox": screen_state.get("next_bbox")} if screen_state.get("next_bbox") else None,
                "cookies": None,
            },
            "actions": actions,
            "screen_state": screen_state,
            "resolved_answer": resolved_answer,
            "fallback_used": bool(fallback_used),
            "page_signature": screen_state.get("page_signature") or (page_data or {}).get("page_signature"),
            "last_action": last_action,
            "last_screen_signature": screen_state.get("screen_signature"),
            "attempt_counts": attempts,
            "transition": transition,
            "controls_meta": (controls_data or {}).get("meta") if isinstance(controls_data, dict) else {},
        }

    def _write_quiz_debug(self, *, screen_state: Dict[str, Any], trace: Dict[str, Any], resolved: Dict[str, Any]) -> None:
        if self.current_run_dir is None:
            return
        try:
            self.current_run_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        parse_path = self.current_run_dir / "screen_quiz_parse.json"
        try:
            parse_payload = {
                "screen_state": screen_state,
                "resolved_answer": resolved,
                "trace": trace,
            }
            parse_path.write_text(json.dumps(parse_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        trace_path = self.current_run_dir / "quiz_trace.jsonl"
        try:
            line = {
                "ts": time.time(),
                "question_text": screen_state.get("question_text"),
                "screen_signature": screen_state.get("screen_signature"),
                "trace": trace,
                "resolved_answer": resolved,
            }
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _quiz_mode_enabled(self) -> bool:
        return str(os.environ.get("FULLBOT_QUIZ_MODE", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}

    def _quiz_answer_cache(self) -> Path:
        env_path = str(os.environ.get("FULLBOT_QUIZ_ANSWER_CACHE", "") or "").strip()
        if env_path:
            return Path(env_path)
        return self.quiz_answer_cache

    def _screen_site_consistency(self) -> Optional[float]:
        return None

    def _log(self, message: str) -> None:
        try:
            self.logger(f"[brain] {message}")
        except Exception:
            pass

    def _error(self, message: str) -> None:
        try:
            self.logger(f"[ERROR] {message}")
        except Exception:
            pass
