from __future__ import annotations

from typing import Any, Dict, List, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def _normalize_targets(values: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in values or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        label_text = str(item.get("labelText") or "").strip()
        aria_label = str(item.get("ariaLabel") or "").strip()
        labelledby_text = str(item.get("labelledbyText") or "").strip()
        chosen_name = text or label_text or aria_label or labelledby_text
        if not chosen_name:
            continue
        source = "text"
        if not text and label_text:
            source = "label"
        elif not text and not label_text and aria_label:
            source = "aria_label"
        elif not text and not label_text and not aria_label and labelledby_text:
            source = "aria_labelledby"
        rows.append(
            {
                "tag": str(item.get("tag") or "").strip().lower(),
                "role": str(item.get("role") or "").strip().lower(),
                "chosen_name": chosen_name,
                "source": source,
                "candidates": {
                    "text": text,
                    "label": label_text,
                    "aria_label": aria_label,
                    "aria_labelledby": labelledby_text,
                },
                "bbox": item.get("bbox"),
            }
        )
    return rows


def build_accessible_name_precedence_contract(
    *,
    current_page_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_targets = _normalize_targets((current_page_state or {}).get("semanticTargets"))
    primary = resolved_targets[0] if resolved_targets else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "has_accessible_name": bool(resolved_targets),
        "reason": "accessible_name_precedence_resolved" if resolved_targets else "no_accessible_name_signal",
        "resolved_count": len(resolved_targets),
        "primary_name": primary,
        "resolved_targets": resolved_targets,
    }
