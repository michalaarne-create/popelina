from __future__ import annotations

from typing import Any, Dict, List, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def _normalize_semantic_targets(values: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in values or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        aria_label = str(item.get("ariaLabel") or "").strip()
        labelledby_text = str(item.get("labelledbyText") or "").strip()
        description = str(item.get("describedbyText") or "").strip()
        role = str(item.get("role") or "").strip().lower()
        tag = str(item.get("tag") or "").strip().lower()
        resolved_name = text or aria_label or labelledby_text
        if not (resolved_name or role or description):
            continue
        source = "text"
        if not text and aria_label:
            source = "aria_label"
        elif not text and not aria_label and labelledby_text:
            source = "aria_labelledby"
        rows.append(
            {
                "role": role or tag or "unknown",
                "tag": tag,
                "resolved_name": resolved_name,
                "text": text,
                "aria_label": aria_label,
                "labelledby_text": labelledby_text,
                "description": description,
                "source": source,
                "bbox": item.get("bbox"),
            }
        )
    return rows


def build_accessibility_identity_resolver(
    *,
    current_page_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    targets = _normalize_semantic_targets((current_page_state or {}).get("semanticTargets"))
    primary = targets[0] if targets else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "has_semantic_identity": bool(targets),
        "reason": "semantic_targets_resolved" if targets else "no_semantic_targets",
        "target_count": len(targets),
        "primary_identity": primary,
        "semantic_targets": targets,
    }
