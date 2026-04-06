from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def _normalize_list(values: Iterable[Any]) -> List[str]:
    items: List[str] = []
    seen = set()
    for value in values:
        text = " ".join(str(value or "").strip().split()).lower()
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
    return items


def build_business_rule_contract(
    *,
    screen_state: Optional[Dict[str, Any]] = None,
    resolved_answer: Optional[Dict[str, Any]] = None,
    page_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    screen = screen_state or {}
    resolved = resolved_answer or {}
    page = page_data or {}
    active_block = screen.get("active_block") if isinstance(screen.get("active_block"), dict) else {}
    dependencies = active_block.get("depends_on") if isinstance(active_block.get("depends_on"), list) else []

    parent_rules: List[Dict[str, Any]] = []
    for row in dependencies:
        if not isinstance(row, dict):
            continue
        parent_rules.append(
            {
                "question_key": str(row.get("question_key") or row.get("qid") or ""),
                "required_values": _normalize_list(row.get("values") or row.get("selected_values") or []),
                "operator": str(row.get("operator") or "any"),
            }
        )

    selected_values = _normalize_list(resolved.get("selected_values") or [])
    visible_qid = str(page.get("qid") or screen.get("question_id") or active_block.get("question_id") or "")
    return {
        "schema_version": SCHEMA_VERSION,
        "question_key": str(active_block.get("question_signature") or screen.get("active_question_signature") or ""),
        "visible_qid": visible_qid,
        "has_conditional_branching": bool(parent_rules),
        "parent_rules": parent_rules,
        "selected_values": selected_values,
        "rule_status": "conditional" if parent_rules else "unconditional",
    }
