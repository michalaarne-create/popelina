"""
Action planner: converts BrainDecision + QA results into low-level click plans.
No side effects here; actual mouse/keyboard control happens elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .types import BBox, BrainDecision, Element


@dataclass
class Action:
    type: str  # "click", "hover", "scroll", "noop"
    target: Optional[Tuple[float, float]] = None
    bbox: Optional[BBox] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    delta: Optional[int] = None  # for scroll


def actions_for_decision(
    decision: BrainDecision,
    selected_option_ids: Optional[List[str]] = None,
    next_element: Optional[Element] = None,
) -> List[Action]:
    actions: List[Action] = []

    if decision.mode == "COOKIES" and decision.target_element:
        actions.append(_click_action(decision.target_element, reason="cookies"))
        return actions

    if decision.mode == "CLICK_NEXT" and decision.target_element:
        actions.append(_click_action(decision.target_element, reason="next"))
        return actions

    if decision.mode == "ANSWER_QUESTION":
        if decision.cluster and selected_option_ids:
            for opt in decision.cluster.options:
                if opt.id in selected_option_ids and opt.bbox:
                    actions.append(Action(type="click", target=_center(opt.bbox), bbox=opt.bbox, meta={"option_id": opt.id}))
            if next_element:
                actions.append(_click_action(next_element, reason="after_answer"))
        else:
            # No QA decision yet -> highlight
            if decision.cluster and decision.cluster.question_bbox:
                actions.append(Action(type="hover", target=_center(decision.cluster.question_bbox), bbox=decision.cluster.question_bbox))
        return actions

    if decision.mode == "SCROLL":
        amount = int(decision.extras.get("amount") or 600)
        direction = str(decision.extras.get("direction") or "down").lower()
        signed = amount if direction != "up" else -amount
        actions.append(Action(type="scroll", delta=signed, meta={"direction": direction, "amount": signed}))
        return actions

    actions.append(Action(type="noop", meta={"reason": decision.reason}))
    return actions


def _click_action(elem: Element, reason: str) -> Action:
    return Action(type="click", target=_center(elem.bbox), bbox=elem.bbox, meta={"element_id": elem.id, "reason": reason})


def _center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
