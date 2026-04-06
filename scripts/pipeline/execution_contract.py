from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence


ActionCallback = Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]
AfterActionCallback = Callable[[Dict[str, Any], int, Dict[str, Any]], Dict[str, Any]]


def execute_action_plan(
    actions_plan: Sequence[Dict[str, Any]],
    *,
    on_screen_click: ActionCallback,
    on_screen_scroll: ActionCallback,
    on_navigate_back: Optional[ActionCallback] = None,
    on_dom_click_button: Optional[ActionCallback] = None,
    on_dom_pick_autocomplete_option: Optional[ActionCallback] = None,
    on_dom_fill_masked_input: Optional[ActionCallback] = None,
    on_key_press: ActionCallback,
    on_key_repeat: ActionCallback,
    on_type_text: ActionCallback,
    on_wait: ActionCallback,
    on_dom_select_option: Optional[ActionCallback] = None,
    on_noop: Optional[ActionCallback] = None,
    after_action: Optional[AfterActionCallback] = None,
) -> List[Dict[str, Any]]:
    executed: List[Dict[str, Any]] = []
    action_index = 0
    for planned in actions_plan:
        if not isinstance(planned, dict):
            continue
        kind = str(planned.get("kind") or "")
        extras: Dict[str, Any] = {}
        if kind == "screen_click":
            extras.update(on_screen_click(planned) or {})
        elif kind == "screen_scroll":
            extras.update(on_screen_scroll(planned) or {})
        elif kind == "navigate_back":
            if not callable(on_navigate_back):
                raise RuntimeError("Unsupported action kind: navigate_back")
            extras.update(on_navigate_back(planned) or {})
        elif kind == "dom_click_button":
            if not callable(on_dom_click_button):
                raise RuntimeError("Unsupported action kind: dom_click_button")
            extras.update(on_dom_click_button(planned) or {})
        elif kind == "dom_pick_autocomplete_option":
            if not callable(on_dom_pick_autocomplete_option):
                raise RuntimeError("Unsupported action kind: dom_pick_autocomplete_option")
            extras.update(on_dom_pick_autocomplete_option(planned) or {})
        elif kind == "dom_fill_masked_input":
            if not callable(on_dom_fill_masked_input):
                raise RuntimeError("Unsupported action kind: dom_fill_masked_input")
            extras.update(on_dom_fill_masked_input(planned) or {})
        elif kind == "key_press":
            extras.update(on_key_press(planned) or {})
        elif kind == "key_repeat":
            extras.update(on_key_repeat(planned) or {})
        elif kind == "type_text":
            extras.update(on_type_text(planned) or {})
        elif kind == "wait":
            extras.update(on_wait(planned) or {})
        elif kind == "dom_select_option":
            if not callable(on_dom_select_option):
                raise RuntimeError("Unsupported action kind: dom_select_option")
            extras.update(on_dom_select_option(planned) or {})
        elif kind == "noop":
            if callable(on_noop):
                extras.update(on_noop(planned) or {})
        else:
            raise RuntimeError(f"Unsupported action kind: {kind}")
        action_index += 1
        post = after_action(planned, action_index, extras) if callable(after_action) else {}
        executed.append({**planned, **extras, **post})
    return executed
