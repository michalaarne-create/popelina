from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    # Central brain-state builder for the screenshot-based pipeline.
    from scripts.pipeline_brain import build_brain_state  # type: ignore
except Exception:  # pragma: no cover - optional optimizer module
    build_brain_state = None  # type: ignore


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
        summary_data = self._load_json(summary_path)
        question_data = self._load_json(self.question_path)
        if summary_data is None:
            self._log(f"Summary missing at {summary_path}; falling back to random.")
            return BrainDecision(
                recommended_action="fallback_random",
                target_bbox=None,
                target_element=None,
                requires_action=True,
                brain_state=self._load_state(),
                question_data=question_data,
                summary_data=None,
            )

        if build_brain_state is None:
            self._log("build_brain_state unavailable; falling back to random.")
            return BrainDecision(
                recommended_action="fallback_random",
                target_bbox=None,
                target_element=None,
                requires_action=True,
                brain_state=self._load_state(),
                question_data=question_data,
                summary_data=summary_data,
            )

        prev_state = self._load_state()
        meta: Dict[str, Any] = {}
        consistency = self._screen_site_consistency()
        if consistency is not None:
            meta = {
                "screen_site_consistency": float(consistency),
                "screen_site_matched": bool(consistency >= 0.5),
            }
            if consistency < 0.5:
                print("Screen and site is not matched. Brain can't decide about question and answers")
                state_dump = dict(prev_state or {})
                state_dump.update(meta)
                self._save_state(state_dump)
                return BrainDecision(
                    recommended_action="idle",
                    target_bbox=None,
                    target_element=None,
                    requires_action=False,
                    brain_state=state_dump,
                    question_data=question_data,
                    summary_data=summary_data,
                )
        try:
            brain = build_brain_state(  # type: ignore[misc]
                question_data,
                summary_data,
                prev_state,
                mark_answer_clicked=False,
                mark_next_clicked=False,
            )
        except Exception as exc:
            self._log(f"build_brain_state failed: {exc}; falling back to random.")
            return BrainDecision(
                recommended_action="fallback_random",
                target_bbox=None,
                target_element=None,
                requires_action=True,
                brain_state=prev_state,
                question_data=question_data,
                summary_data=summary_data,
            )
        if not isinstance(brain, dict):
            self._log("build_brain_state returned non-dict; falling back to random.")
            return BrainDecision(
                recommended_action="fallback_random",
                target_bbox=None,
                target_element=None,
                requires_action=True,
                brain_state=prev_state,
                question_data=question_data,
                summary_data=summary_data,
            )
        action = brain.get("recommended_action", "idle")
        target_element, target_bbox = self._select_target(brain, action)

        # Update simple per-question flags (answer/next). Cookies are stateless for now.
        mark_answer = action == "click_answer" and target_bbox is not None
        mark_next = action == "click_next" and target_bbox is not None
        updated = build_brain_state(
            question_data,
            summary_data,
            prev_state,
            mark_answer_clicked=mark_answer,
            mark_next_clicked=mark_next,
        )
        self._save_state(updated)

        # Require an actual agent action for these high-level intents.
        requires_action = action in {
            "click_answer",
            "click_next",
            "click_cookies_accept",
            "scroll_page_down",
        }
        return BrainDecision(
            recommended_action=action,
            target_bbox=target_bbox,
            target_element=target_element,
            requires_action=requires_action,
            brain_state=updated,
            question_data=question_data,
            summary_data=summary_data,
        )

    def load_state(self) -> dict:
        return self._load_state()

    def _load_json(self, path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self._log(f"Could not read JSON {path}.")
            return None

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            self._log(f"Could not read brain state {self.state_path}.")
            return {}

    def _save_state(self, state: dict) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self._log(f"Failed to save brain state: {exc}")

    def _select_target(
        self,
        brain: dict,
        action: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[List[float]]]:
        objects = brain.get("objects") or {}
        target: Optional[Dict[str, Any]] = None

        if action == "click_answer":
            answers = objects.get("answers") or []
            target = answers[0] if answers else None
        elif action == "click_next":
            target = objects.get("next")
        elif action == "click_cookies_accept":
            target = objects.get("cookies")

        if not target:
            return None, None

        bbox = target.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                coords = [float(c) for c in bbox]
                return target, coords
            except Exception:
                return target, None
        return target, None

    def _log(self, message: str) -> None:
        try:
            self.logger(f"[brain] {message}")
        except Exception:
            pass
