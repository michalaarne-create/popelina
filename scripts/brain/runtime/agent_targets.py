from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _coerce_bbox(target: Optional[Dict[str, Any]]) -> Optional[List[float]]:
    if not isinstance(target, dict):
        return None
    bbox = target.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        return [float(c) for c in bbox]
    except Exception:
        return None


def _first_target_with_bbox(items: Any) -> Tuple[Optional[Dict[str, Any]], Optional[List[float]]]:
    if not isinstance(items, list):
        return None, None
    fallback_target: Optional[Dict[str, Any]] = None
    for item in items:
        if not isinstance(item, dict):
            continue
        if fallback_target is None:
            fallback_target = item
        bbox = _coerce_bbox(item)
        if bbox is not None:
            return item, bbox
    return fallback_target, _coerce_bbox(fallback_target)


def select_target(
    brain: Dict[str, Any],
    action: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[List[float]]]:
    objects = brain.get("objects") or {}

    if action == "click_answer":
        return _first_target_with_bbox(objects.get("answers") or [])
    elif action == "click_next":
        target = objects.get("next")
    elif action == "click_cookies_accept":
        target = objects.get("cookies")
    else:
        target = None

    if not target:
        return None, None

    return target, _coerce_bbox(target)
