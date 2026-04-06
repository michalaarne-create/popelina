from __future__ import annotations

from typing import Any, Dict, List, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def _normalize_semantic_targets(values: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in values or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        aria_checked = str(item.get("ariaChecked") or "").strip().lower()
        aria_expanded = str(item.get("ariaExpanded") or "").strip().lower()
        aria_invalid = str(item.get("ariaInvalid") or "").strip().lower()
        if not (role or aria_checked or aria_expanded or aria_invalid):
            continue
        rows.append(
            {
                "role": role or "unknown",
                "aria_checked": aria_checked,
                "aria_expanded": aria_expanded,
                "aria_invalid": aria_invalid,
                "resolved_name": str(item.get("text") or item.get("ariaLabel") or item.get("labelledbyText") or "").strip(),
                "bbox": item.get("bbox"),
            }
        )
    return rows


def build_aria_signal_registry(
    *,
    current_page_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    signals = _normalize_semantic_targets((current_page_state or {}).get("semanticTargets"))
    primary = signals[0] if signals else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "has_aria_signals": bool(signals),
        "reason": "aria_signals_registered" if signals else "no_aria_signals",
        "signal_count": len(signals),
        "primary_signal": primary,
        "signals": signals,
    }
