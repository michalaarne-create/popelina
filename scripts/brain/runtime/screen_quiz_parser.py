from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .canonical_type_contract import build_canonical_type_contract, canonical_operational_type
from .confidence_fusion_policy import build_confidence_fusion_policy
from .control_kind_classifier import classify_control_kind
from .element_role_classifier import classify_element_roles
from .question_block_builder import (
    build_blocks_view as _build_blocks_view_impl,
    build_question_blocks as _build_question_blocks_impl,
    choose_active_question as _choose_active_question_impl,
    rescue_empty_dropdown_block as _rescue_empty_dropdown_block_impl,
)
from .question_summary_bridge import (
    build_summary_backed_question,
    enrich_active_question_from_summary,
)
from .quiz_ai_inputs import has_structural_prompt_tokens, items_have_structural_tokens
from .quiz_type_classifier import classify_quiz_type
from .screen_item_pipeline import (
    build_option_objects as _build_option_objects_impl,
    candidate_type as _candidate_type_impl,
    content_column as _content_column_impl,
    dedupe_items as _dedupe_items_impl,
    filter_to_content as _filter_to_content_impl,
    option_horiz_overlap_ratio as _option_horiz_overlap_ratio_impl,
    prepare_items as _prepare_items_impl,
    prompt_candidates as _prompt_candidates_impl,
    prune_option_candidates as _prune_option_candidates_impl,
)
from .summary_candidate_extractor import (
    rated_answer_candidates as _rated_answer_candidates_impl,
    summary_answer_candidates as _summary_answer_candidates_impl,
    summary_dropdown_candidates as _summary_dropdown_candidates_impl,
    summary_first_bbox as _summary_first_bbox_impl,
    summary_question_candidate as _summary_question_candidate_impl,
)
from .quiz_utils import (
    box_center,
    box_height,
    box_iou,
    box_union,
    box_width,
    clean_option_text,
    clamp_float,
    header_like,
    md5_text,
    next_like,
    normalize_match_text,
    normalize_ocr_text,
    normalize_question_text,
    question_like,
    sha1_text,
    text_similarity,
)


_PAGE_HEADER_TOKENS = (
    "home",
    "test quiz server",
    "radio +",
    "checkbox",
    "dropdown",
    "select(",
    "select ->",
    "input + next",
    "jednokrotna odpowied",
    "wielokrotna odpowied",
)
_MULTI_HINT_TOKENS = ("zaznacz", "wybierz wszystkie", "wielokrot")
_SCROLL_HINT_TOKENS = ("scroll", "przew", "właściwą opcję")
_TRIPLE_HINT_TOKENS = ("(1/3)", "(2/3)", "(3/3)", "1/3", "2/3", "3/3")
_MIX_HINT_TOKENS = ("(mix)", "mix)")
_PROMPT_HINT_TOKENS = (
    "zaznacz",
    "wybierz",
    "wybierz wszystkie",
    "wpisz",
    "uzupelnij",
    "uzupełnij",
    "podaj",
    "jaki",
    "jaka",
    "jakie",
    "ktory",
    "ktora",
    "ktore",
    "który",
    "która",
    "które",
    "ile",
    "dopasuj",
)
def _input_placeholder_like(text: str) -> bool:
    norm = normalize_match_text(text)
    if not norm:
        return False
    return norm.startswith("wpisz odpowied")


def _prompt_prefers_text(text: str) -> bool:
    norm = normalize_match_text(text)
    if not norm:
        return False
    return any(tok in norm for tok in ("wpisz", "podaj", "uzupelnij", "uzupełnij"))


def _prompt_prefers_choice(text: str) -> bool:
    norm = normalize_match_text(text)
    if not norm:
        return False
    choice_tokens = (
        "wybierz",
        "zaznacz",
        "choose one",
        "one option",
        "jedna odpowiedz",
        "co najmniej",
        "wszystkie",
    )
    return any(tok in norm for tok in choice_tokens)


def _bbox4(value: Any) -> Optional[List[int]]:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _extract_box(item: Dict[str, Any]) -> Optional[List[int]]:
    for key in ("text_box", "dropdown_box", "bbox"):
        value = item.get(key)
        if isinstance(value, list) and len(value) == 4:
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in value]
            except Exception:
                continue
            if x2 > x1 and y2 > y1:
                return [x1, y1, x2, y2]
        if isinstance(value, dict):
            try:
                x = int(round(float(value.get("x", 0))))
                y = int(round(float(value.get("y", 0))))
                w = int(round(float(value.get("width", 0))))
                h = int(round(float(value.get("height", 0))))
            except Exception:
                continue
            if w > 0 and h > 0:
                return [x, y, x + w, y + h]
    return None


def _image_size(payload: Dict[str, Any]) -> Tuple[int, int]:
    width = 0
    height = 0
    for row in payload.get("results") or []:
        box = _extract_box(row) if isinstance(row, dict) else None
        if box:
            width = max(width, box[2])
            height = max(height, box[3])
    return max(width, 1920), max(height, 1080)


def _dedupe_items(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _dedupe_items_impl(items)


def _prepare_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _prepare_items_impl(payload, extract_box_fn=_extract_box)


def _page_header_like(text: str) -> bool:
    norm = normalize_match_text(text)
    if not norm:
        return False
    if any(token in norm for token in _PAGE_HEADER_TOKENS):
        return True
    if ("pytanie" in norm) and any(token in norm for token in ("next", "auto", "select", "dropdown", "radio", "checkbox", "input")):
        return True
    return False


def _looks_like_instruction(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if _page_header_like(raw):
        return False
    if _input_placeholder_like(raw):
        return False
    norm = normalize_match_text(raw)
    if not norm:
        return False
    if question_like(raw):
        return True
    has_hint = any(tok in norm for tok in _PROMPT_HINT_TOKENS) or any(tok in norm for tok in _MULTI_HINT_TOKENS)
    if has_hint and (raw.endswith(":") or ("?" in raw) or len(norm) >= 18):
        return True
    if ("?" in raw) and len(norm) >= 8:
        return True
    return False


def _looks_like_option_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if _page_header_like(raw) or next_like(raw):
        return False
    if _looks_like_instruction(raw):
        return False
    norm = normalize_match_text(raw)
    if not norm:
        return False
    words = [w for w in norm.split() if w]
    if len(raw) <= 64 and len(words) <= 7 and ("?" not in raw) and (":" not in raw):
        return True
    return False


def _option_horiz_overlap_ratio(box: Sequence[int], content_column: Sequence[int]) -> float:
    return _option_horiz_overlap_ratio_impl(box, content_column)


def _prune_option_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    content_column: Sequence[int],
    question_bbox: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    return _prune_option_candidates_impl(
        candidates,
        content_column=content_column,
        question_bbox=question_bbox,
        bbox4_fn=_bbox4,
    )


def _content_column(items: Sequence[Dict[str, Any]], screen_w: int) -> List[int]:
    return _content_column_impl(items, screen_w)


def _candidate_type(item: Dict[str, Any], screen_h: int) -> str:
    return _candidate_type_impl(
        item,
        screen_h=screen_h,
        page_header_like_fn=_page_header_like,
        header_like_fn=header_like,
        looks_like_instruction_fn=_looks_like_instruction,
        next_like_fn=next_like,
    )


def _filter_to_content(items: Sequence[Dict[str, Any]], content_column: Sequence[int], screen_h: int) -> List[Dict[str, Any]]:
    return _filter_to_content_impl(
        items,
        content_column_value=content_column,
        screen_h=screen_h,
        candidate_type_fn=_candidate_type,
    )


def _prompt_candidates(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _prompt_candidates_impl(
        items,
        input_placeholder_like_fn=_input_placeholder_like,
        page_header_like_fn=_page_header_like,
        looks_like_instruction_fn=_looks_like_instruction,
        looks_like_option_text_fn=_looks_like_option_text,
        next_like_fn=next_like,
    )


def _build_option_objects(items: Sequence[Dict[str, Any]], question_box: Sequence[int], next_y: int) -> List[Dict[str, Any]]:
    return _build_option_objects_impl(
        items,
        question_box=question_box,
        next_y=next_y,
        looks_like_instruction_fn=_looks_like_instruction,
        page_header_like_fn=_page_header_like,
        next_like_fn=next_like,
    )


def _summary_answer_candidates(summary_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _summary_answer_candidates_impl(
        summary_data,
        bbox4_fn=_bbox4,
        looks_like_instruction_fn=_looks_like_instruction,
        page_header_like_fn=_page_header_like,
        next_like_fn=next_like,
    )


def _summary_question_candidate(summary_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return _summary_question_candidate_impl(
        summary_data,
        bbox4_fn=_bbox4,
        looks_like_instruction_fn=_looks_like_instruction,
    )


def _rated_answer_candidates(
    rated_data: Optional[Dict[str, Any]],
    *,
    question_text: str = "",
    question_bbox: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    return _rated_answer_candidates_impl(
        rated_data,
        question_text=question_text,
        question_bbox=question_bbox,
        bbox4_fn=_bbox4,
        looks_like_instruction_fn=_looks_like_instruction,
        page_header_like_fn=_page_header_like,
        next_like_fn=next_like,
    )


def _summary_first_bbox(summary_data: Optional[Dict[str, Any]], *labels: str) -> Optional[List[int]]:
    return _summary_first_bbox_impl(
        summary_data,
        *labels,
        bbox4_fn=_bbox4,
        page_header_like_fn=_page_header_like,
    )


def _summary_dropdown_candidates(summary_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _summary_dropdown_candidates_impl(
        summary_data,
        bbox4_fn=_bbox4,
        page_header_like_fn=_page_header_like,
    )


def _rescue_empty_dropdown_block(
    active: Optional[Dict[str, Any]],
    *,
    summary_data: Optional[Dict[str, Any]],
    content_column: Sequence[int],
    screen_h: int,
) -> None:
    _rescue_empty_dropdown_block_impl(
        active,
        summary_data=summary_data,
        content_column=content_column,
        screen_h=screen_h,
        summary_dropdown_candidates_fn=_summary_dropdown_candidates,
        prompt_prefers_text_fn=_prompt_prefers_text,
        bbox4_fn=_bbox4,
    )


def _build_question_blocks(items: Sequence[Dict[str, Any]], screen_h: int, content_column: Sequence[int]) -> List[Dict[str, Any]]:
    return _build_question_blocks_impl(
        items,
        screen_h=screen_h,
        content_column=content_column,
        prompt_candidates_fn=_prompt_candidates,
        build_option_objects_fn=_build_option_objects,
        page_header_like_fn=_page_header_like,
        input_placeholder_like_fn=_input_placeholder_like,
        prompt_prefers_text_fn=_prompt_prefers_text,
        prompt_prefers_choice_fn=_prompt_prefers_choice,
        looks_like_instruction_fn=_looks_like_instruction,
    )


def _serialize_element_roles(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role_probs = item.get("role_probs") if isinstance(item.get("role_probs"), dict) else {}
        role_probs_small = {
            str(key): round(float(value), 4)
            for key, value in role_probs.items()
            if float(value or 0.0) >= 0.08
        }
        out.append(
            {
                "id": str(item.get("id") or ""),
                "text": str(item.get("text") or ""),
                "bbox": [int(v) for v in (item.get("bbox") or [0, 0, 0, 0])],
                "candidate_type": str(item.get("candidate_type") or ""),
                "role_pred": str(item.get("role_pred") or ""),
                "role_conf": round(float(item.get("role_conf") or 0.0), 4),
                "role_probs": role_probs_small,
                "has_frame": bool(item.get("has_frame")),
            }
        )
    return out


def _build_blocks_view(questions: Sequence[Dict[str, Any]], active_id: Optional[str]) -> List[Dict[str, Any]]:
    return _build_blocks_view_impl(questions, active_id)


def _choose_active_question(questions: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return _choose_active_question_impl(questions, page_header_like_fn=_page_header_like)


def _detect_screen_quiz_type(active: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(active, dict):
        return {
            "detected_quiz_type": "single",
            "detected_operational_type": "choice",
            "type_confidence": 0.0,
            "type_source": "screen",
            "type_signals": {"reason": "no_active_question"},
        }

    control_kind = str(active.get("control_kind") or "unknown")
    prompt_norm = str(active.get("prompt_norm") or "")
    options = active.get("options") or []
    signals: Dict[str, Any] = {"control_kind": control_kind, "options_count": len(options) if isinstance(options, list) else 0}
    option_norms = [str((opt or {}).get("norm_text") or "") for opt in options if isinstance(opt, dict)]
    joined_options = "\n".join(text for text in option_norms if text)
    matrix_prompt_hint = any(tok in prompt_norm for tok in ("tabel", "siatk", "macierz", "ocen wiersze", "oceń wiersze"))
    matrix_option_hint = any(("wiersz" in text and "kolumn" in text) for text in option_norms)
    matrix_grid_hint = bool(option_norms) and joined_options.count("/") >= max(1, min(3, len(option_norms)))

    def _ret(qtype: str, conf: float) -> Dict[str, Any]:
        op = canonical_operational_type(qtype)
        return {
            "detected_quiz_type": qtype,
            "detected_operational_type": op,
            "type_confidence": float(round(clamp_float(conf, 0.0, 1.0), 4)),
            "type_source": "screen",
            "type_signals": signals,
        }

    if control_kind == "text":
        signals["rule"] = "control_kind=text"
        return _ret("text", 0.97)

    if control_kind == "dropdown":
        has_scroll_hint = any(tok in prompt_norm for tok in _SCROLL_HINT_TOKENS)
        signals["has_scroll_hint"] = bool(has_scroll_hint)
        signals["rule"] = "control_kind=dropdown"
        return _ret("dropdown_scroll" if has_scroll_hint else "dropdown", 0.93 if has_scroll_hint else 0.9)

    if matrix_prompt_hint or matrix_option_hint or matrix_grid_hint:
        signals["matrix_prompt_hint"] = bool(matrix_prompt_hint)
        signals["matrix_option_hint"] = bool(matrix_option_hint)
        signals["matrix_grid_hint"] = bool(matrix_grid_hint)
        signals["rule"] = "matrix_prompt_or_option_pattern"
        return _ret("matrix", 0.85 if matrix_prompt_hint or matrix_option_hint else 0.78)

    if any(tok in prompt_norm for tok in _MIX_HINT_TOKENS):
        signals["rule"] = "prompt_mix_token"
        return _ret("mixed", 0.86)
    if any(tok in prompt_norm for tok in _TRIPLE_HINT_TOKENS):
        signals["rule"] = "prompt_triple_token"
        return _ret("triple", 0.9)
    if any(tok in prompt_norm for tok in _MULTI_HINT_TOKENS):
        signals["rule"] = "prompt_multi_token"
        return _ret("multi", 0.74)

    signals["rule"] = "default_choice_single"
    return _ret("single", 0.66)


def _build_question_split_payload(
    *,
    active: Optional[Dict[str, Any]],
    questions: Sequence[Dict[str, Any]],
    detected_quiz_type: str,
) -> Dict[str, Any]:
    active = active if isinstance(active, dict) else {}
    active_options = active.get("options") if isinstance(active.get("options"), list) else []
    question_text = str(active.get("prompt_text") or "")
    split: Dict[str, Any] = {
        "question": question_text,
        "answers": [str((o or {}).get("text") or "") for o in active_options if isinstance(o, dict)],
    }
    if detected_quiz_type not in {"mixed", "triple"}:
        return split

    parts: List[Dict[str, Any]] = []
    for idx, question in enumerate(questions):
        if not isinstance(question, dict):
            continue
        prompt_text = str(question.get("prompt_text") or "").strip()
        if not prompt_text:
            continue
        options = question.get("options") if isinstance(question.get("options"), list) else []
        part = {
            "id": str(question.get("id") or f"part_{idx}"),
            "question": prompt_text,
            "control_kind": str(question.get("control_kind") or "unknown"),
            "answers": [str((o or {}).get("text") or "") for o in options if isinstance(o, dict)],
            "has_input": bool(question.get("input_bbox")),
            "has_select": bool(question.get("select_bbox")),
        }
        parts.append(part)

    if parts:
        split["parts"] = parts
        split["answers"] = [str(part.get("question") or "") for part in parts if str(part.get("question") or "")]
        if not split["question"]:
            split["question"] = str(parts[0].get("question") or "")
    return split


def parse_screen_quiz_state(
    *,
    region_payload: Optional[Dict[str, Any]],
    summary_data: Optional[Dict[str, Any]] = None,
    page_data: Optional[Dict[str, Any]] = None,
    rated_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = region_payload or {}
    screen_w, screen_h = _image_size(payload)
    items = _prepare_items(payload)
    content_column = _content_column(items, screen_w)
    filtered = _filter_to_content(items, content_column, screen_h)
    filtered = classify_element_roles(filtered, screen_w, screen_h)
    questions = _build_question_blocks(filtered, screen_h, content_column)
    active = _choose_active_question(questions) if questions else None
    _rescue_empty_dropdown_block(active, summary_data=summary_data, content_column=content_column, screen_h=screen_h)
    question_hint = _summary_question_candidate(summary_data)
    question_hint_text = str((question_hint or {}).get("text") or "")
    question_hint_bbox = _bbox4((question_hint or {}).get("bbox")) if isinstance(question_hint, dict) else None
    summary_answer_candidates = _summary_answer_candidates(summary_data)
    if not summary_answer_candidates:
        summary_answer_candidates = _rated_answer_candidates(
            rated_data,
            question_text=question_hint_text,
            question_bbox=question_hint_bbox,
        )
    summary_answer_candidates = _prune_option_candidates(
        summary_answer_candidates,
        content_column=content_column,
        question_bbox=question_hint_bbox,
    )
    if (
        isinstance(question_hint, dict)
        and len(questions) >= 2
        and (
            (not isinstance(active, dict))
            or _page_header_like(str(active.get("prompt_text") or ""))
        )
        and _looks_like_instruction(question_hint_text)
        and (
            not has_structural_prompt_tokens(
                question_hint_text,
                prompt_triple=_TRIPLE_HINT_TOKENS,
                prompt_mix=_MIX_HINT_TOKENS,
            )
        )
        and (
            not items_have_structural_tokens(
                filtered,
                summary_data=summary_data,
                prompt_triple=_TRIPLE_HINT_TOKENS,
                prompt_mix=_MIX_HINT_TOKENS,
            )
        )
    ):
        collapsed = build_summary_backed_question(
            question_hint,
            summary_data=summary_data,
            summary_answer_candidates=summary_answer_candidates,
            screen_w=screen_w,
            screen_h=screen_h,
            content_column=content_column,
            scroll_hint_tokens=_SCROLL_HINT_TOKENS,
            summary_first_bbox_fn=_summary_first_bbox,
            prompt_prefers_text_fn=_prompt_prefers_text,
            prompt_prefers_choice_fn=_prompt_prefers_choice,
            bbox4_fn=_bbox4,
            mode="collapse",
        )
        questions = [collapsed]
        active = collapsed
    if (not active) and isinstance(question_hint, dict):
        active = build_summary_backed_question(
            question_hint,
            summary_data=summary_data,
            summary_answer_candidates=summary_answer_candidates,
            screen_w=screen_w,
            screen_h=screen_h,
            content_column=content_column,
            scroll_hint_tokens=_SCROLL_HINT_TOKENS,
            summary_first_bbox_fn=_summary_first_bbox,
            prompt_prefers_text_fn=_prompt_prefers_text,
            prompt_prefers_choice_fn=_prompt_prefers_choice,
            bbox4_fn=_bbox4,
            mode="active",
        )
    if active:
        enrich_active_question_from_summary(
            active,
            summary_data=summary_data,
            summary_answer_candidates=summary_answer_candidates,
            screen_h=screen_h,
            content_column=content_column,
            summary_first_bbox_fn=_summary_first_bbox,
            prune_option_candidates_fn=_prune_option_candidates,
            bbox4_fn=_bbox4,
        )
    try:
        type_detection = classify_quiz_type(
            region_payload=payload if isinstance(payload, dict) else None,
            summary_data=summary_data if isinstance(summary_data, dict) else None,
            rated_data=rated_data if isinstance(rated_data, dict) else None,
            questions=questions,
            active=active,
            screen_w=screen_w,
            screen_h=screen_h,
        )
    except Exception:
        type_detection = _detect_screen_quiz_type(active)
        type_detection["decision_margin"] = 0.0
        type_detection["type_probs"] = {
            "single": 1.0 if type_detection.get("detected_quiz_type") == "single" else 0.0,
            "multi": 1.0 if type_detection.get("detected_quiz_type") == "multi" else 0.0,
            "dropdown": 1.0 if type_detection.get("detected_quiz_type") == "dropdown" else 0.0,
            "dropdown_scroll": 1.0 if type_detection.get("detected_quiz_type") == "dropdown_scroll" else 0.0,
            "text": 1.0 if type_detection.get("detected_quiz_type") == "text" else 0.0,
            "triple": 1.0 if type_detection.get("detected_quiz_type") == "triple" else 0.0,
            "mixed": 1.0 if type_detection.get("detected_quiz_type") == "mixed" else 0.0,
            "unknown": 0.0,
        }
        type_detection["quiz_type_features"] = {}
        type_detection["question_split"] = _build_question_split_payload(
            active=active,
            questions=questions,
            detected_quiz_type=str(type_detection.get("detected_quiz_type") or ""),
        )
    base_control_kind = str(active.get("control_kind") or "unknown") if isinstance(active, dict) else "unknown"
    provisional_state = {
        "questions": questions,
        "blocks": _build_blocks_view(questions, active.get("id") if isinstance(active, dict) else None),
        "active_block": None,
        "question_text": active.get("prompt_text") if isinstance(active, dict) else "",
        "options": (active.get("options") or []) if isinstance(active, dict) else [],
        "control_kind": base_control_kind,
        "detected_quiz_type": type_detection.get("detected_quiz_type"),
        "active_block_type": str(type_detection.get("active_block_type") or ""),
        "type_confidence": type_detection.get("type_confidence"),
        "decision_margin": float(type_detection.get("decision_margin") or 0.0),
        "type_signals": type_detection.get("type_signals") if isinstance(type_detection.get("type_signals"), dict) else {},
        "quiz_type_features": type_detection.get("quiz_type_features") if isinstance(type_detection.get("quiz_type_features"), dict) else {},
        "next_bbox": active.get("next_bbox") if isinstance(active, dict) else None,
        "select_bbox": active.get("select_bbox") if isinstance(active, dict) else None,
        "input_bbox": active.get("input_bbox") if isinstance(active, dict) else None,
        "scroll_needed": bool(active.get("scroll_needed")) if isinstance(active, dict) else False,
    }
    block_types = type_detection.get("block_types") if isinstance(type_detection.get("block_types"), list) else []
    if block_types:
        for idx, q in enumerate(questions):
            if not isinstance(q, dict):
                continue
            if idx < len(block_types):
                q["block_type"] = str(block_types[idx] or "")
    active_id = active.get("id") if isinstance(active, dict) else None
    blocks = _build_blocks_view(questions, active_id)
    active_block = next((block for block in blocks if block.get("is_active")), None)
    provisional_state["blocks"] = blocks
    provisional_state["active_block"] = active_block
    control_model = classify_control_kind(provisional_state)
    predicted_control_kind = str(control_model.get("pred") or base_control_kind)
    predicted_control_conf = float(control_model.get("conf") or 0.0)
    if isinstance(active, dict) and predicted_control_conf >= 0.84 and predicted_control_kind in {"choice", "dropdown", "text", "slider"}:
        current_quiz_type = str(type_detection.get("detected_quiz_type") or "")
        current_active_block_type = str(type_detection.get("active_block_type") or "")
        changed = predicted_control_kind != base_control_kind
        if changed:
            active["control_kind"] = predicted_control_kind
            if isinstance(active_block, dict):
                active_block["control_kind"] = predicted_control_kind
                if predicted_control_kind == "dropdown":
                    active_block["block_family"] = "dropdown"
                    if current_active_block_type in {"single", "multi", "unknown", ""}:
                        type_detection["active_block_type"] = "dropdown_scroll" if "scroll" in current_quiz_type else "dropdown"
                elif predicted_control_kind == "text":
                    active_block["block_family"] = "text"
                    if current_active_block_type in {"single", "multi", "unknown", ""}:
                        type_detection["active_block_type"] = "text"
                elif predicted_control_kind == "slider":
                    active_block["block_family"] = "slider"
                    # Keep explicit matrix blocks stable even if the control classifier sees slider-like artifacts.
                    if current_active_block_type in {"single", "multi", "unknown", ""}:
                        type_detection["active_block_type"] = "slider"
            if predicted_control_kind == "dropdown" and current_quiz_type in {"single", "multi"}:
                type_detection["detected_quiz_type"] = "dropdown_scroll" if bool(active.get("scroll_needed")) or str(current_quiz_type).endswith("scroll") or bool((type_detection.get("type_signals") or {}).get("artifact_prompt_flags", {}).get("scroll_hint")) else "dropdown"
                type_detection["detected_operational_type"] = canonical_operational_type(type_detection["detected_quiz_type"])
            elif predicted_control_kind == "text" and current_quiz_type in {"single", "multi", "dropdown", "dropdown_scroll"}:
                type_detection["detected_quiz_type"] = "text"
                type_detection["detected_operational_type"] = canonical_operational_type(type_detection["detected_quiz_type"])
            elif predicted_control_kind == "slider" and current_quiz_type in {"single", "multi", "dropdown", "dropdown_scroll", "text"}:
                type_detection["detected_quiz_type"] = "slider"
                type_detection["detected_operational_type"] = canonical_operational_type(type_detection["detected_quiz_type"])
        if predicted_control_kind == "choice" and str(type_detection.get("detected_operational_type") or "") == "unknown":
            type_detection["detected_operational_type"] = canonical_operational_type("choice")
    screen_type_detection = _detect_screen_quiz_type(active)
    if (
        isinstance(active, dict)
        and str(screen_type_detection.get("detected_quiz_type") or "") == "matrix"
        and str(type_detection.get("detected_quiz_type") or "") != "matrix"
    ):
        type_detection["detected_quiz_type"] = "matrix"
        type_detection["detected_operational_type"] = "matrix"
        type_detection["type_confidence"] = max(
            float(type_detection.get("type_confidence") or 0.0),
            float(screen_type_detection.get("type_confidence") or 0.0),
        )
        merged_signals = dict(type_detection.get("type_signals") or {})
        merged_signals.update(screen_type_detection.get("type_signals") or {})
        merged_signals["matrix_guard_override"] = True
        type_detection["type_signals"] = merged_signals
        active["control_kind"] = "matrix"
        if isinstance(active_block, dict):
            active_block["control_kind"] = "matrix"
            active_block["block_family"] = "matrix"
        if str(type_detection.get("active_block_type") or "") in {"", "unknown", "single", "multi", "slider"}:
            type_detection["active_block_type"] = "matrix"
    element_roles = _serialize_element_roles(filtered)
    texts_for_signature = [q.get("prompt_norm") or "" for q in questions]
    if active:
        texts_for_signature.extend(opt.get("norm_text") or "" for opt in active.get("options") or [])
        texts_for_signature.append(active.get("control_kind") or "")
    screen_signature = md5_text("\n".join(texts_for_signature) or "__empty_screen__")
    page_signature = None
    if isinstance(page_data, dict):
        page_signature = str(page_data.get("page_signature") or "") or None
    confidence = {
        "screen": round(
            clamp_float(
                (
                    float(active.get("confidence") or 0.0)
                    + min(1.0, len(active.get("options") or []) / 4.0 if active else 0.0)
                )
                / 2.0,
                0.0,
                1.0,
            ),
            4,
        ) if active else 0.0,
        "merged": 0.0,
        "dom": 0.0,
    }
    confidence["dom"] = 0.75 if isinstance(page_data, dict) and page_data else 0.0
    confidence_policy = build_confidence_fusion_policy(
        screen_confidence=float(confidence["screen"] or 0.0),
        type_confidence=float(type_detection.get("type_confidence") or 0.0),
        decision_margin=float(type_detection.get("decision_margin") or 0.0),
        control_model=control_model if isinstance(control_model, dict) else {},
        has_dom_context=bool(page_data or active),
    )
    confidence["merged"] = float(confidence_policy.get("fused_confidence") or confidence["screen"])
    parse_quality = "empty"
    if active:
        if active.get("next_bbox") or active.get("select_bbox") or active.get("input_bbox") or active.get("options"):
            parse_quality = "actionable"
        else:
            parse_quality = "question_only"
    return {
        "page_signature": page_signature,
        "screen_signature": screen_signature,
        "questions": questions,
        "blocks": blocks,
        "active_block": active_block,
        "active_block_id": active_block.get("id") if isinstance(active_block, dict) else None,
        "active_block_index": active_block.get("block_index") if isinstance(active_block, dict) else None,
        "active_question_id": active.get("id") if active else None,
        "active_question_signature": active.get("question_signature") if active else None,
        "question_text": active.get("prompt_text") if active else "",
        "options": (active.get("options") or []) if active else [],
        "control_kind": active.get("control_kind") if active else "unknown",
        "control_kind_model": control_model,
        "confidence_fusion_policy": confidence_policy,
        "detected_quiz_type": type_detection.get("detected_quiz_type"),
        "detected_operational_type": type_detection.get("detected_operational_type"),
        "canonical_type_contract": build_canonical_type_contract(
            detected_quiz_type=str(type_detection.get("detected_quiz_type") or ""),
            control_kind=active.get("control_kind") if active else "unknown",
            active_block_type=str(type_detection.get("active_block_type") or ""),
            source="screen_parser",
        ),
        "layout_type": str(type_detection.get("layout_type") or ("multi_block" if len(questions) >= 2 else "single_block")),
        "active_block_type": str(type_detection.get("active_block_type") or ""),
        "top_global_type": str(type_detection.get("top_global_type") or ""),
        "type_confidence": type_detection.get("type_confidence"),
        "type_source": type_detection.get("type_source"),
        "type_signals": type_detection.get("type_signals"),
        "type_probs": type_detection.get("type_probs") if isinstance(type_detection.get("type_probs"), dict) else {},
        "decision_margin": float(type_detection.get("decision_margin") or 0.0),
        "type_reason": str(type_detection.get("type_reason") or ""),
        "quiz_type_features": type_detection.get("quiz_type_features") if isinstance(type_detection.get("quiz_type_features"), dict) else {},
        "question_split": type_detection.get("question_split") if isinstance(type_detection.get("question_split"), dict) else {},
        "parse_signature_v2": str(type_detection.get("parse_signature_v2") or ""),
        "block_types": type_detection.get("block_types") if isinstance(type_detection.get("block_types"), list) else [],
        "element_roles": element_roles,
        "input_bbox": active.get("input_bbox") if active else None,
        "select_bbox": active.get("select_bbox") if active else None,
        "next_bbox": active.get("next_bbox") if active else None,
        "scroll_needed": bool(active.get("scroll_needed")) if active else False,
        "screen_targets": {
            "answer_candidates": (active.get("options") or []) if active else [],
            "next_bbox": active.get("next_bbox") if active else None,
            "select_bbox": active.get("select_bbox") if active else None,
            "input_bbox": active.get("input_bbox") if active else None,
        },
        "screen_parse_quality": parse_quality,
        "content_column": [int(content_column[0]), 0, int(content_column[2]), int(screen_h)],
        "confidence": confidence,
        "image": payload.get("image"),
        "summary_source": (summary_data or {}).get("_source"),
        "parser_debug": {
            "screen_size": {"width": screen_w, "height": screen_h},
            "items_considered": len(items),
            "items_filtered": len(filtered),
            "blocks_count": len(blocks),
            "active_block_id": active_block.get("id") if isinstance(active_block, dict) else None,
            "content_column": [int(content_column[0]), 0, int(content_column[2]), int(screen_h)],
            "parse_quality": parse_quality,
        },
    }
