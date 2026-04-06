from __future__ import annotations

from typing import Any, Dict, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def build_hard_stop_guard_state(
    *,
    submit_guard_state: Optional[Dict[str, Any]] = None,
    anti_modal_guard: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    submit_guard = submit_guard_state or {}
    anti_modal = anti_modal_guard or {}

    if bool(submit_guard.get("is_hard_stop")):
        reason = str(submit_guard.get("stop_reason") or "challenge_detected")
        return {
            "schema_version": SCHEMA_VERSION,
            "is_blocking": True,
            "source": "submit_guard",
            "reason": reason,
            "error": f"Safe stop before submit: {reason}",
            "signals": {
                "submit_guard_hard_stop": True,
                "anti_modal_blocking": bool(anti_modal.get("is_blocking")),
            },
        }

    if bool(anti_modal.get("is_blocking")):
        return {
            "schema_version": SCHEMA_VERSION,
            "is_blocking": True,
            "source": "anti_modal_guard",
            "reason": str(anti_modal.get("reason") or "unexpected_modal_overlay"),
            "error": "Blocked by anti modal guard",
            "signals": {
                "submit_guard_hard_stop": False,
                "anti_modal_blocking": True,
            },
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "is_blocking": False,
        "source": "none",
        "reason": "clear",
        "error": "",
        "signals": {
            "submit_guard_hard_stop": False,
            "anti_modal_blocking": False,
        },
    }
