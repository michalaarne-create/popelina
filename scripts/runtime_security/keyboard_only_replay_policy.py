from __future__ import annotations

from typing import Any, Dict, Sequence

from scripts.pipeline.contracts import SCHEMA_VERSION

from .keyboard_action_policy import build_keyboard_action_policy


_KEYBOARD_ONLY_KINDS = {"key_press", "key_repeat", "type_text", "wait", "noop"}


def build_keyboard_only_replay_policy(
    *,
    actions_plan: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    actions = list(actions_plan or [])
    keyboard_policy = build_keyboard_action_policy(actions_plan=actions)
    first_non_keyboard = None

    for index, action in enumerate(actions):
        kind = str((action or {}).get("kind") or "").strip().lower()
        if kind not in _KEYBOARD_ONLY_KINDS:
            first_non_keyboard = {
                "index": index,
                "kind": kind,
                "reason": str((action or {}).get("reason") or "").strip().lower(),
            }
            break

    replayable = first_non_keyboard is None and bool(keyboard_policy.get("is_allowed"))
    reason = "keyboard_only_replay_available"
    if first_non_keyboard is not None:
        reason = "mouse_or_dom_action_present"
    elif not bool(keyboard_policy.get("is_allowed")):
        reason = str(keyboard_policy.get("reason") or "unsafe_keyboard_action")

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "allow" if replayable else "block",
        "reason": reason,
        "is_replayable": replayable,
        "requires_review": not replayable,
        "keyboard_action_policy": keyboard_policy,
        "first_non_keyboard_action": first_non_keyboard,
    }
