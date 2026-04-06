from __future__ import annotations

from typing import Any, Dict, List, Optional

from .canonical_type_contract import canonical_operational_type


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def infer_quiz_type_from_controls(
    controls_data: Optional[Dict[str, Any]],
    *,
    screen_state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(controls_data, dict):
        return None
    controls = controls_data.get("controls") if isinstance(controls_data.get("controls"), list) else []
    kinds = {str((row or {}).get("kind") or "").strip().lower() for row in controls if isinstance(row, dict)}
    if not kinds:
        return None
    prompt = _normalize_text((screen_state or {}).get("question_text") or "")
    qid = str(((controls_data.get("meta") or {}).get("qid") or "")).strip().lower()
    qtype = None
    if "textbox" in kinds:
        qtype = "text"
    elif "select" in kinds:
        qtype = "dropdown_scroll" if "scroll" in prompt else "dropdown"
    elif "checkbox" in kinds:
        qtype = "multi"
    elif "radio" in kinds:
        qtype = "single"
    if qtype is None:
        return None
    if qid.startswith("type09_") or qid.startswith("type10_"):
        qtype = "triple"
    elif qid.startswith("type13_"):
        qtype = "mixed"
    return {
        "detected_quiz_type": qtype,
        "detected_operational_type": canonical_operational_type(qtype),
        "type_confidence": 0.95,
        "type_source": "dom_fallback",
        "type_signals": {
            "controls_kinds": sorted(kinds),
            "qid": qid,
        },
    }


def build_dom_action_plan(
    *,
    page_state: Dict[str, Any],
    cache_item: Dict[str, Any],
    preferred_action: str = "",
) -> Optional[Dict[str, Any]]:
    if not isinstance(page_state, dict) or not isinstance(cache_item, dict):
        return None
    preferred = str(preferred_action or "").strip().lower()
    if preferred == "click_back" and bool(page_state.get("backVisible")):
        return {
            "source": "page_state_navigation",
            "action_kind": "back",
            "target": {"label": str(page_state.get("backLabel") or "").strip()},
            "reason": "click_back",
        }
    if preferred == "save_draft" and bool(page_state.get("draftVisible")):
        return {
            "source": "page_state_navigation",
            "action_kind": "save_draft",
            "target": {"label": str(page_state.get("draftLabel") or "").strip()},
            "reason": "save_draft",
        }
    if not any(bool(page_state.get(key)) for key in ("hasTextbox", "hasSelect")) and not page_state.get("options"):
        return None

    qtype = str(cache_item.get("question_type") or "").strip().lower()
    operational_qtype = canonical_operational_type(qtype, fallback=qtype or "choice")
    correct_answer = str(cache_item.get("text_answer") or cache_item.get("correct_answer") or "").strip()
    selected = [str(v) for v in (cache_item.get("selected_options") or [])]
    options_text = cache_item.get("options_text") if isinstance(cache_item.get("options_text"), dict) else {}

    answers: List[str] = []
    if operational_qtype == "text":
        if correct_answer:
            answers = [correct_answer]
    elif selected and options_text:
        answers = [str(options_text.get(key) or "").strip() for key in selected if str(options_text.get(key) or "").strip()]
    elif correct_answer:
        answers = [correct_answer]

    action_kind = "choice"
    reason = "choice_answer"
    if operational_qtype == "autocomplete" and bool(page_state.get("hasTextbox")):
        action_kind = "autocomplete"
        reason = "autocomplete_answer"
    elif operational_qtype == "masked_input" and bool(page_state.get("hasTextbox")):
        action_kind = "masked_input"
        reason = "masked_input_answer"
    elif bool(page_state.get("hasTextbox")):
        action_kind = "text"
        reason = "textbox_answer"
    elif bool(page_state.get("hasSelect")):
        action_kind = "select"
        reason = "select_answer"

    return {
        "source": "dom_fallback",
        "action_kind": action_kind,
        "target": {"answers": answers},
        "reason": reason,
    }
