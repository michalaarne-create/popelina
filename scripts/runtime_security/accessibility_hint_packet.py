from __future__ import annotations

from typing import Any, Dict, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def build_accessibility_hint_packet(
    *,
    accessibility_identity_resolver: Optional[Dict[str, Any]] = None,
    accessible_name_precedence_contract: Optional[Dict[str, Any]] = None,
    aria_signal_registry: Optional[Dict[str, Any]] = None,
    status_role_interpreter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    identity = accessibility_identity_resolver or {}
    name_contract = accessible_name_precedence_contract or {}
    aria_registry = aria_signal_registry or {}
    status_role = status_role_interpreter or {}

    primary_identity = identity.get("primary_identity") or {}
    primary_name = name_contract.get("primary_name") or {}
    primary_signal = aria_registry.get("primary_signal") or {}

    return {
        "schema_version": SCHEMA_VERSION,
        "has_hint": any(
            (
                bool(identity.get("has_semantic_identity")),
                bool(name_contract.get("has_accessible_name")),
                bool(aria_registry.get("has_aria_signals")),
                bool(status_role.get("is_eventful")),
            )
        ),
        "reason": "accessibility_hint_ready",
        "hint": {
            "role": str(primary_signal.get("role") or primary_identity.get("role") or ""),
            "name": str(primary_name.get("chosen_name") or primary_identity.get("resolved_name") or ""),
            "name_source": str(primary_name.get("source") or primary_identity.get("source") or ""),
            "description": str(primary_identity.get("description") or ""),
            "aria_checked": str(primary_signal.get("aria_checked") or ""),
            "aria_expanded": str(primary_signal.get("aria_expanded") or ""),
            "aria_invalid": str(primary_signal.get("aria_invalid") or ""),
            "status_category": str(status_role.get("category") or "none"),
        },
        "signals": {
            "identity_available": bool(identity.get("has_semantic_identity")),
            "name_available": bool(name_contract.get("has_accessible_name")),
            "aria_available": bool(aria_registry.get("has_aria_signals")),
            "status_available": bool(status_role.get("is_eventful")),
        },
    }
