from __future__ import annotations

from typing import Any, Dict, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def _has_valid_bbox(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 4


def build_focus_visibility_splitter(
    *,
    current_page_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cur = current_page_state or {}
    textbox_focused = bool(cur.get("textboxFocused"))
    textbox_visible = bool(cur.get("hasTextbox")) and _has_valid_bbox(cur.get("textboxBbox"))
    select_visible = bool(cur.get("hasSelect")) and _has_valid_bbox(cur.get("selectBbox"))

    focus_target = "textbox" if textbox_focused else "none"
    visual_target = "textbox" if textbox_visible else ("select" if select_visible else "none")
    focus_without_visual = focus_target != "none" and focus_target != visual_target

    reason = "focus_matches_visibility"
    if focus_target == "textbox" and not textbox_visible:
        reason = "focused_textbox_not_visually_interactive"
    elif focus_target == "none" and visual_target != "none":
        reason = "visible_control_without_focus"

    return {
        "schema_version": SCHEMA_VERSION,
        "focus_target": focus_target,
        "visual_target": visual_target,
        "focus_without_visual": focus_without_visual,
        "is_split": focus_without_visual,
        "reason": reason,
        "signals": {
            "textbox_focused": textbox_focused,
            "textbox_visible": textbox_visible,
            "select_visible": select_visible,
        },
    }
