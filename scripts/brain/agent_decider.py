from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .agent_state_io import load_json, load_state, save_state
from .agent_targets import select_target
from .pipeline_state_builder import build_brain_state


def _default_logger(message: str) -> None:
    print(message)


@dataclass
class BrainDecision:
    recommended_action: str
    target_bbox: Optional[List[float]]
    target_element: Optional[Dict[str, Any]]
    requires_action: bool
    brain_state: dict
    question_data: Optional[dict]
    summary_data: Optional[dict]


class PipelineBrainAgent:
    def __init__(
        self,
        question_path: Path,
        state_path: Path,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.question_path = question_path
        self.state_path = state_path
        self.logger = logger or _default_logger

    def decide(self, summary_path: Path) -> BrainDecision:
        summary_data = load_json(summary_path, self._log)
        question_data = load_json(self.question_path, self._log)
        prev_state = load_state(self.state_path, self._log)
        if summary_data is None:
            self._log(f"Summary missing at {summary_path}; falling back to random.")
            return BrainDecision("fallback_random", None, None, True, prev_state, question_data, None)

        consistency = self._screen_site_consistency()
        if consistency is not None and consistency < 0.5:
            print("Screen and site is not matched. Brain can't decide about question and answers")
            state_dump = dict(prev_state or {})
            state_dump.update({"screen_site_consistency": float(consistency), "screen_site_matched": False})
            save_state(self.state_path, state_dump, self._log)
            return BrainDecision("idle", None, None, False, state_dump, question_data, summary_data)

        try:
            brain = build_brain_state(question_data, summary_data, prev_state, False, False)
        except Exception as exc:
            self._log(f"build_brain_state failed: {exc}; falling back to random.")
            return BrainDecision("fallback_random", None, None, True, prev_state, question_data, summary_data)
        if not isinstance(brain, dict):
            self._log("build_brain_state returned non-dict; falling back to random.")
            return BrainDecision("fallback_random", None, None, True, prev_state, question_data, summary_data)

        action = brain.get("recommended_action", "idle")
        target_element, target_bbox = select_target(brain, action)
        mark_answer = action == "click_answer" and target_bbox is not None
        mark_next = action == "click_next" and target_bbox is not None
        updated = build_brain_state(question_data, summary_data, prev_state, mark_answer, mark_next)
        save_state(self.state_path, updated, self._log)
        requires_action = action in {"click_answer", "click_next", "click_cookies_accept", "scroll_page_down"}
        return BrainDecision(action, target_bbox, target_element, requires_action, updated, question_data, summary_data)

    def load_state(self) -> dict:
        return load_state(self.state_path, self._log)

    def _screen_site_consistency(self) -> Optional[float]:
        # Legacy placeholder kept for compatibility; when unavailable, no gating is applied.
        return None

    def _log(self, message: str) -> None:
        try:
            self.logger(f"[brain] {message}")
        except Exception:
            pass

