from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def select_target(
    brain: Dict[str, Any],
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

