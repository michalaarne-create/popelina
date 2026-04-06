from __future__ import annotations

from typing import Any, Dict, List, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION

from .data_redaction import redact_text


def _redact_string_list(values: Any) -> List[str]:
    redacted: List[str] = []
    for item in values or []:
        text = str(item or "")
        redacted.append(redact_text(text))
    return redacted


def redact_expected_values_for_storage(values: Any) -> List[str]:
    return _redact_string_list(values)


def redact_actions_for_storage(actions: Any) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for action in actions or []:
        if not isinstance(action, dict):
            continue
        row = dict(action)
        if "text" in row:
            row["text"] = redact_text(row.get("text"))
        if "value" in row:
            row["value"] = redact_text(row.get("value"))
        sanitized.append(row)
    return sanitized


def redact_resolved_answer_for_storage(resolved_answer: Any) -> Dict[str, Any]:
    if not isinstance(resolved_answer, dict):
        return {}
    row = dict(resolved_answer)
    for key in ("correct_answers", "selected_values", "expected_values"):
        if key in row:
            row[key] = _redact_string_list(row.get(key))
    return row


def redact_trace_for_storage(trace: Any) -> Dict[str, Any]:
    if not isinstance(trace, dict):
        return {}
    row = dict(trace)
    post_action_expectation = dict(row.get("post_action_expectation") or {})
    if post_action_expectation:
        if "expected_values" in post_action_expectation:
            post_action_expectation["expected_values"] = _redact_string_list(post_action_expectation.get("expected_values"))
        row["post_action_expectation"] = post_action_expectation
    return row


def redact_page_state_for_storage(page_state: Any) -> Dict[str, Any]:
    if not isinstance(page_state, dict):
        return {}
    row = dict(page_state)
    if "textboxValue" in row:
        row["textboxValue"] = redact_text(row.get("textboxValue"))
    return row


def redact_screen_plan_for_storage(screen_plan: Any) -> Dict[str, Any]:
    if not isinstance(screen_plan, dict):
        return {}
    row = dict(screen_plan)
    if "target" in row and isinstance(row.get("target"), dict):
        target = dict(row.get("target") or {})
        if "answers" in target:
            target["answers"] = _redact_string_list(target.get("answers"))
        if "target_text" in target:
            target["target_text"] = redact_text(target.get("target_text"))
        row["target"] = target
    return row


def redact_previous_action_meta_for_storage(previous_action: Any) -> Dict[str, Any]:
    if not isinstance(previous_action, dict):
        return {}
    row = dict(previous_action)
    if "target_text" in row:
        row["target_text"] = redact_text(row.get("target_text"))
    if "expected_values" in row:
        row["expected_values"] = _redact_string_list(row.get("expected_values"))
    if "post_action_expectation" in row and isinstance(row.get("post_action_expectation"), dict):
        expectation = dict(row.get("post_action_expectation") or {})
        if "expected_values" in expectation:
            expectation["expected_values"] = _redact_string_list(expectation.get("expected_values"))
        row["post_action_expectation"] = expectation
    return row


def build_trace_payload_redaction_policy(
    *,
    actions: Any,
    resolved_answer: Any,
    trace: Any,
    page_state: Any,
) -> Dict[str, Any]:
    redacted_action_fields = sum(1 for action in actions or [] if isinstance(action, dict) and any(key in action for key in ("text", "value")))
    redacted_answer_fields = sum(1 for key in ("correct_answers", "selected_values", "expected_values") if isinstance(resolved_answer, dict) and key in resolved_answer)
    redacted_trace_fields = 1 if isinstance(((trace or {}).get("post_action_expectation") if isinstance(trace, dict) else {}), dict) and "expected_values" in ((trace or {}).get("post_action_expectation") or {}) else 0
    redacted_page_state_fields = 1 if isinstance(page_state, dict) and "textboxValue" in page_state else 0
    return {
        "schema_version": SCHEMA_VERSION,
        "redacted_action_fields": int(redacted_action_fields),
        "redacted_answer_fields": int(redacted_answer_fields),
        "redacted_trace_fields": int(redacted_trace_fields),
        "redacted_page_state_fields": int(redacted_page_state_fields),
        "has_redactions": bool(redacted_action_fields or redacted_answer_fields or redacted_trace_fields or redacted_page_state_fields),
    }
