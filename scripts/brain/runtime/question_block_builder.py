from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .question_grouping_model import score_item_for_prompt
from .screen_item_pipeline import merge_multi_column_options
from .quiz_utils import box_height, box_iou, box_union, box_width, clamp_float, normalize_match_text, normalize_question_text, sha1_text


def guess_next_bbox(
    section_items: Sequence[Dict[str, Any]],
    *,
    section_end_y: int,
    page_header_like_fn: Callable[[str], bool],
    looks_like_instruction_fn: Callable[[str], bool],
    input_placeholder_like_fn: Callable[[str], bool],
) -> Optional[List[int]]:
    role_matches = [
        item["bbox"]
        for item in section_items
        if str(item.get("role_pred") or "") == "next"
        and float(item.get("role_conf") or 0.0) >= 0.45
    ]
    if role_matches:
        return [int(v) for v in role_matches[0]]
    text_matches = [
        item["bbox"]
        for item in section_items
        if item.get("candidate_type") == "next_button"
    ]
    if text_matches:
        return [int(v) for v in text_matches[0]]
    buttonish = []
    for item in section_items:
        box = item["bbox"]
        if box_height(box) < 18 or box_height(box) > 70:
            continue
        if box_width(box) < 40 or box_width(box) > 260:
            continue
        if box[1] < int(section_end_y * 0.65):
            continue
        text = normalize_match_text(item.get("text") or "")
        if not text or len(text) > 14 or page_header_like_fn(text):
            continue
        if looks_like_instruction_fn(text) or input_placeholder_like_fn(text):
            continue
        buttonish.append(item)
    if buttonish:
        buttonish.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        return [int(v) for v in buttonish[0]["bbox"]]
    return None


def build_question_blocks(
    items: Sequence[Dict[str, Any]],
    *,
    screen_h: int,
    content_column: Sequence[int],
    prompt_candidates_fn: Callable[[Sequence[Dict[str, Any]]], List[Dict[str, Any]]],
    build_option_objects_fn: Callable[[Sequence[Dict[str, Any]], Sequence[int], int], List[Dict[str, Any]]],
    page_header_like_fn: Callable[[str], bool],
    input_placeholder_like_fn: Callable[[str], bool],
    prompt_prefers_text_fn: Callable[[str], bool],
    prompt_prefers_choice_fn: Callable[[str], bool],
    looks_like_instruction_fn: Callable[[str], bool],
) -> List[Dict[str, Any]]:
    prompts = prompt_candidates_fn(items)
    questions: List[Dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        box = prompt["bbox"]
        next_prompt_y = prompts[idx + 1]["bbox"][1] if idx + 1 < len(prompts) else screen_h + 1
        section_items = [
            item
            for item in items
            if item["bbox"][1] >= box[1] and item["bbox"][1] < next_prompt_y
        ]
        framed = [
            item
            for item in section_items
            if item.get("candidate_type") == "dropdown_trigger" and item["bbox"][1] >= box[3]
        ]
        next_bbox = guess_next_bbox(
            section_items,
            section_end_y=next_prompt_y,
            page_header_like_fn=page_header_like_fn,
            looks_like_instruction_fn=looks_like_instruction_fn,
            input_placeholder_like_fn=input_placeholder_like_fn,
        )
        next_y = next_bbox[1] if next_bbox else next_prompt_y
        options = build_option_objects_fn(section_items, box, next_y)
        if options:
            source_items = {str(item.get("id") or ""): item for item in section_items if isinstance(item, dict)}
            prompt_for_model = {
                "id": prompt.get("id"),
                "text": prompt.get("text"),
                "prompt_text": prompt.get("text"),
                "bbox": box,
                "control_kind": "choice",
            }
            rescored: List[Dict[str, Any]] = []
            for opt in options:
                item = source_items.get(str(opt.get("id") or ""))
                opt = dict(opt)
                if not isinstance(item, dict):
                    rescored.append(opt)
                    continue
                prob = score_item_for_prompt(prompt_for_model, item, screen_h)
                opt["grouping_prob"] = round(float(prob), 4) if prob is not None else None
                rescored.append(opt)
            strong_grouped = [
                opt
                for opt in rescored
                if opt.get("grouping_prob") is not None and float(opt.get("grouping_prob") or 0.0) >= 0.35
            ]
            if strong_grouped:
                rescored = strong_grouped
            rescored.sort(
                key=lambda opt: -float(opt.get("grouping_prob") if opt.get("grouping_prob") is not None else -1.0)
            )
            options = merge_multi_column_options(rescored)
        control_kind = "choice"
        select_bbox = None
        input_bbox = None
        prompt_text = str(prompt.get("text") or "")
        prompt_norm = normalize_match_text(prompt_text)
        prompt_dropdown_box = (
            prompt.get("dropdown_box")
            if isinstance(prompt.get("dropdown_box"), list) and len(prompt.get("dropdown_box")) == 4
            else None
        )
        prompt_frame_is_control = bool(
            prompt_dropdown_box
            and box_width(prompt_dropdown_box) >= (box_width(box) * 1.35)
            and prompt_dropdown_box[3] >= (box[3] + 8)
            and ("?" not in prompt_text)
        )
        framed_non_placeholder = [
            item for item in framed if not input_placeholder_like_fn(item.get("text") or "")
        ]
        if prompt_frame_is_control and not prompt_prefers_text_fn(prompt_text):
            select_bbox = [int(v) for v in (prompt_dropdown_box or box)]
            control_kind = "dropdown"
        elif framed_non_placeholder and not prompt_prefers_text_fn(prompt_text):
            frame_box = framed_non_placeholder[0].get("dropdown_box") or framed_non_placeholder[0]["bbox"]
            select_bbox = [int(v) for v in frame_box]
            control_kind = "dropdown"
        elif options and ("scroll" in prompt_norm) and len(options) >= 6 and not prompt_prefers_text_fn(prompt_text):
            ox1 = min(int(opt["bbox"][0]) for opt in options)
            oy1 = min(int(opt["bbox"][1]) for opt in options)
            ox2 = max(int(opt["bbox"][2]) for opt in options)
            oy2 = max(int(opt["bbox"][3]) for opt in options)
            select_bbox = [
                max(int(content_column[0]), ox1 - 24),
                max(int(box[3] + 8), oy1 - 18),
                min(int(content_column[2]), ox2 + 220),
                min(int(next_y - 8), oy2 + 18),
            ]
            control_kind = "dropdown"
        elif not options:
            control_kind = "text"
            input_bbox = [
                int(content_column[0]),
                int(box[3] + max(14, box_height(box) * 0.7)),
                int(content_column[2]),
                int((next_bbox[1] - 12) if next_bbox else min(screen_h - 10, box[3] + 88)),
            ]
            if input_bbox[3] <= input_bbox[1]:
                input_bbox[3] = input_bbox[1] + 44
        if control_kind == "dropdown" and len(options) >= 3 and prompt_prefers_choice_fn(prompt_text) and ("scroll" not in prompt_norm):
            control_kind = "choice"
            select_bbox = None
        question_signature = sha1_text(
            normalize_match_text(prompt["text"]) + "\n" + "\n".join(opt["norm_text"] for opt in options)
        )
        cluster_box = box_union([box] + [opt["bbox"] for opt in options] + ([next_bbox] if next_bbox else []))
        questions.append(
            {
                "id": f"screen_q_{idx}",
                "question_signature": question_signature,
                "prompt_text": normalize_question_text(prompt["text"]),
                "prompt_norm": normalize_match_text(prompt["text"]),
                "bbox": [int(v) for v in box],
                "cluster_bbox": cluster_box or [int(v) for v in box],
                "control_kind": control_kind,
                "options": options,
                "next_bbox": next_bbox,
                "select_bbox": select_bbox,
                "input_bbox": input_bbox,
                "scroll_needed": bool((cluster_box or box)[3] >= int(screen_h * 0.92)),
                "confidence": round(clamp_float(float(prompt.get("conf") or 0.0), 0.0, 1.0), 4),
            }
        )
        if select_bbox and ("scroll" in prompt_norm):
            questions[-1]["scroll_needed"] = True
            questions[-1]["block_type"] = "dropdown_scroll"
    return questions


def rescue_empty_dropdown_block(
    active: Optional[Dict[str, Any]],
    *,
    summary_data: Optional[Dict[str, Any]],
    content_column: Sequence[int],
    screen_h: int,
    summary_dropdown_candidates_fn: Callable[[Optional[Dict[str, Any]]], List[Dict[str, Any]]],
    prompt_prefers_text_fn: Callable[[str], bool],
    bbox4_fn: Callable[[Any], Optional[List[int]]],
) -> None:
    if not isinstance(active, dict):
        return
    if str(active.get("control_kind") or "") != "text":
        return
    if active.get("options") or active.get("select_bbox") or (not active.get("next_bbox")):
        return
    prompt_text = str(active.get("prompt_text") or "")
    prompt_norm = normalize_match_text(prompt_text)
    if not prompt_norm or prompt_prefers_text_fn(prompt_text):
        return
    prompt_box = bbox4_fn(active.get("bbox"))
    next_box = bbox4_fn(active.get("next_bbox"))
    if prompt_box is None or next_box is None:
        return
    candidates = summary_dropdown_candidates_fn(summary_data)
    match_found = False
    for candidate in candidates:
        cand_box = candidate["bbox"]
        same_prompt = candidate.get("norm_text") == prompt_norm
        overlaps_prompt = box_iou(cand_box, prompt_box) >= 0.55
        close_y = abs(int(cand_box[1]) - int(prompt_box[1])) <= 36
        if same_prompt or (overlaps_prompt and close_y):
            match_found = True
            break
    if not match_found:
        return
    x1 = int(content_column[0])
    x2 = int(content_column[2])
    y1 = int(max(prompt_box[3] + 10, prompt_box[1] + max(24, box_height(prompt_box))))
    y2 = int(min(next_box[1] - 12, y1 + max(42, int(screen_h * 0.08))))
    if y2 <= y1:
        y2 = min(screen_h - 8, y1 + 44)
    if x2 <= x1:
        return
    active["control_kind"] = "dropdown"
    active["block_type"] = "dropdown_scroll" if ("scroll" in prompt_norm) else "dropdown"
    active["select_bbox"] = [x1, y1, x2, y2]
    active["input_bbox"] = None
    active["scroll_needed"] = bool("scroll" in prompt_norm)


def block_family(block: Dict[str, Any]) -> str:
    control_kind = str(block.get("control_kind") or "unknown")
    block_type = str(block.get("block_type") or "")
    if block_type in {"single", "multi", "dropdown", "dropdown_scroll", "text", "triple", "mixed", "slider"}:
        return block_type
    if control_kind == "text":
        return "text"
    if control_kind == "dropdown":
        if block_type == "dropdown_scroll":
            return "dropdown_scroll"
        return "dropdown"
    return "choice"


def build_blocks_view(questions: Sequence[Dict[str, Any]], active_id: Optional[str]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for idx, question in enumerate(questions):
        if not isinstance(question, dict):
            continue
        block_id = str(question.get("id") or f"screen_q_{idx}")
        options = question.get("options") if isinstance(question.get("options"), list) else []
        block_type = str(question.get("block_type") or "")
        blocks.append(
            {
                "id": block_id,
                "block_index": idx,
                "question_signature": str(question.get("question_signature") or ""),
                "prompt_text": str(question.get("prompt_text") or ""),
                "prompt_norm": str(question.get("prompt_norm") or ""),
                "bbox": [int(v) for v in (question.get("bbox") or [0, 0, 0, 0])],
                "cluster_bbox": [int(v) for v in (question.get("cluster_bbox") or question.get("bbox") or [0, 0, 0, 0])],
                "control_kind": str(question.get("control_kind") or "unknown"),
                "block_type": block_type,
                "block_family": block_family(question),
                "answer_count": len(options),
                "answers": options,
                "has_next": bool(question.get("next_bbox")),
                "has_select": bool(question.get("select_bbox")),
                "has_input": bool(question.get("input_bbox")),
                "next_bbox": question.get("next_bbox"),
                "select_bbox": question.get("select_bbox"),
                "input_bbox": question.get("input_bbox"),
                "scroll_needed": bool(question.get("scroll_needed")),
                "confidence": round(float(question.get("confidence") or 0.0), 4),
                "is_active": block_id == active_id,
            }
        )
    return blocks


def choose_active_question(
    questions: Sequence[Dict[str, Any]],
    *,
    page_header_like_fn: Callable[[str], bool],
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, float, float]] = None
    fallback_best: Optional[Dict[str, Any]] = None
    fallback_key: Optional[Tuple[float, float, float]] = None
    for idx, question in enumerate(questions):
        if not isinstance(question, dict):
            continue
        prompt_text = str(question.get("prompt_text") or "")
        options = question.get("options") if isinstance(question.get("options"), list) else []
        already_answered = bool(
            question.get("already_answered")
            or question.get("is_answered")
            or question.get("solved")
        )
        in_viewport = question.get("in_viewport")
        if in_viewport is None:
            in_viewport = question.get("is_in_viewport")
        viewport_ok = True if in_viewport is None else bool(in_viewport)
        score = 0.0
        score += min(4.0, float(len(options)))
        if question.get("select_bbox"):
            score += 2.5
        if question.get("next_bbox"):
            score += 1.5
        if question.get("input_bbox"):
            score += 0.8
        if bool(question.get("scroll_needed")):
            score += 1.0
        if page_header_like_fn(prompt_text):
            score -= 4.0
        box = question.get("bbox") if isinstance(question.get("bbox"), list) else [0, 0, 0, 0]
        y = float(box[1]) if len(box) == 4 else float(idx * 1000)
        conf = float(question.get("confidence") or 0.0)
        key = (score, conf, -y)
        if fallback_key is None or key > fallback_key:
            fallback_best = question
            fallback_key = key
        if already_answered or not viewport_ok:
            continue
        if best_key is None or key > best_key:
            best = question
            best_key = key
    return best or fallback_best
