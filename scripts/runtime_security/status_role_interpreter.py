from __future__ import annotations

from typing import Any, Dict, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def _normalize_texts(values: Any) -> list[str]:
    texts: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text:
            texts.append(text)
    return texts


def build_status_role_interpreter(
    *,
    current_page_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state = current_page_state or {}
    status_texts = _normalize_texts(state.get("statusRoleTexts"))
    alert_texts = _normalize_texts(state.get("alertRoleTexts"))
    dialog_texts = _normalize_texts(state.get("dialogRoleTexts"))
    progress_values = _normalize_texts(state.get("progressbarValues"))

    category = "none"
    reason = "no_status_role_signal"
    if alert_texts:
        category = "alert"
        reason = "alert_role_detected"
    elif dialog_texts:
        category = "dialog"
        reason = "dialog_role_detected"
    elif progress_values:
        category = "progressbar"
        reason = "progressbar_role_detected"
    elif status_texts:
        category = "status"
        reason = "status_role_detected"

    return {
        "schema_version": SCHEMA_VERSION,
        "category": category,
        "reason": reason,
        "is_eventful": category != "none",
        "signals": {
            "status_count": len(status_texts),
            "alert_count": len(alert_texts),
            "dialog_count": len(dialog_texts),
            "progressbar_count": len(progress_values),
        },
        "role_texts": {
            "status": status_texts,
            "alert": alert_texts,
            "dialog": dialog_texts,
            "progressbar": progress_values,
        },
    }
