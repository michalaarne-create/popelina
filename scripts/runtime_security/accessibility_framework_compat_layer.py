from __future__ import annotations

from typing import Any, Dict, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def _normalize_texts(*values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for group in values:
        for value in group or []:
            text = str(value or "").strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                result.append(text)
    return result


def build_accessibility_framework_compat_layer(
    *,
    current_page_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state = current_page_state or {}
    status_texts = _normalize_texts(
        state.get("statusRoleTexts"),
        state.get("statusTexts"),
        state.get("statusMessages"),
    )
    alert_texts = _normalize_texts(
        state.get("alertRoleTexts"),
        state.get("alertTexts"),
        state.get("alertMessages"),
    )
    dialog_texts = _normalize_texts(
        state.get("dialogRoleTexts"),
        state.get("dialogTexts"),
        state.get("modalTexts"),
    )
    progressbar_values = _normalize_texts(
        state.get("progressbarValues"),
        state.get("progressValues"),
        state.get("progressTexts"),
    )
    live_region_texts = _normalize_texts(
        state.get("liveRegionTexts"),
        state.get("ariaLiveTexts"),
        state.get("liveMessages"),
    )
    textbox_aria_invalid = bool(
        state.get("textboxAriaInvalid")
        or state.get("ariaInvalid")
        or state.get("fieldAriaInvalid")
    )

    alias_hits = sum(
        1
        for key in (
            "statusTexts",
            "statusMessages",
            "alertTexts",
            "alertMessages",
            "dialogTexts",
            "modalTexts",
            "progressValues",
            "progressTexts",
            "ariaLiveTexts",
            "liveMessages",
            "ariaInvalid",
            "fieldAriaInvalid",
        )
        if key in state
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "is_compatible": True,
        "reason": "accessibility_signals_normalized",
        "normalized_contract": {
            "status_role_texts": status_texts,
            "alert_role_texts": alert_texts,
            "dialog_role_texts": dialog_texts,
            "progressbar_values": progressbar_values,
            "live_region_texts": live_region_texts,
            "textbox_aria_invalid": textbox_aria_invalid,
        },
        "signals": {
            "alias_hits": alias_hits,
            "normalized_signal_count": sum(
                len(values)
                for values in (
                    status_texts,
                    alert_texts,
                    dialog_texts,
                    progressbar_values,
                    live_region_texts,
                )
            ) + int(textbox_aria_invalid),
        },
    }
