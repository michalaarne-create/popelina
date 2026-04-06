from __future__ import annotations

from typing import Any, Dict, List, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION

from .data_redaction import redact_text

_TEXT_ACTION_KINDS = {"type_text", "dom_fill_masked_input", "dom_pick_autocomplete_option"}


def build_entered_data_cleanup_contract(
    *,
    steps: Optional[List[Dict[str, Any]]] = None,
    privacy_reset_contract: Optional[Dict[str, Any]] = None,
    test_data_governance_packet: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entered_payloads: List[str] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        for action in step.get("actions") or []:
            if not isinstance(action, dict):
                continue
            kind = str(action.get("kind") or "").strip().lower()
            if kind not in _TEXT_ACTION_KINDS:
                continue
            text = str(action.get("text") or "").strip()
            if text:
                entered_payloads.append(text)

    redacted_payloads: List[str] = []
    for payload in entered_payloads:
        redacted = redact_text(payload)
        if redacted not in redacted_payloads:
            redacted_payloads.append(redacted)

    reset_signals = (privacy_reset_contract or {}).get("signals") if isinstance(privacy_reset_contract, dict) else {}
    reset_ready = bool((reset_signals or {}).get("cookies_cleared")) and bool((reset_signals or {}).get("storage_cleared"))
    environment = str(((test_data_governance_packet or {}).get("environment")) or "").strip().lower()
    environment_profile = (
        dict((test_data_governance_packet or {}).get("environment_profile") or {})
        if isinstance(test_data_governance_packet, dict)
        else {}
    )

    if not redacted_payloads:
        return {
            "schema_version": SCHEMA_VERSION,
            "requires_cleanup": False,
            "is_ready": True,
            "reason": "no_entered_payloads",
            "cleanup_strategy": "none",
            "typed_payload_count": 0,
            "entered_payloads": [],
            "environment": environment or "default",
            "environment_profile": environment_profile,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "requires_cleanup": True,
        "is_ready": reset_ready,
        "reason": "privacy_reset_ready" if reset_ready else "privacy_reset_not_ready",
        "cleanup_strategy": "privacy_reset_on_session_end" if reset_ready else "manual_cleanup_required",
        "typed_payload_count": int(len(redacted_payloads)),
        "entered_payloads": redacted_payloads,
        "environment": environment or "default",
        "environment_profile": environment_profile,
    }
