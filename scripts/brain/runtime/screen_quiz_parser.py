from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    deduped: List[Dict[str, Any]] = []
    for item in items:
        merged = False
        for existing in deduped:
            same_text = existing.get("norm_text") == item.get("norm_text")
            close = box_iou(existing.get("bbox"), item.get("bbox")) >= 0.55
            similar = text_similarity(existing.get("text"), item.get("text")) >= 0.96
            if close and (same_text or similar):
                if float(item.get("conf") or 0.0) > float(existing.get("conf") or 0.0):
                    existing.update(item)
                merged = True
                break
        if not merged:
            deduped.append(dict(item))
    deduped.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return deduped


def _prepare_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for idx, raw in enumerate(payload.get("results") or []):
        if not isinstance(raw, dict):
            continue
        box = _extract_box(raw)
        if box is None:
            continue
        text = normalize_ocr_text(raw.get("text") or raw.get("box_text") or "")
        if not text:
            continue
        items.append(
            {
                "id": str(raw.get("id") or f"rg_{idx}"),
                "text": text,
                "norm_text": normalize_match_text(text),
                "bbox": box,
                "conf": float(raw.get("conf") or 0.0),
                "has_frame": bool(raw.get("has_frame")),
                "dropdown_box": raw.get("dropdown_box") if isinstance(raw.get("dropdown_box"), list) else None,
                "raw": raw,
            }
        )
    items.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return _dedupe_items(items)


def _content_column(items: Sequence[Dict[str, Any]], screen_w: int) -> List[int]:
    if not items:
        return [0, 0, screen_w, 0]
    bins = max(8, min(48, int(screen_w / 40)))
    scores = [0.0 for _ in range(bins)]
    for item in items:
        box = item["bbox"]
        x1 = max(0, min(bins - 1, int((box[0] / max(1, screen_w)) * bins)))
        x2 = max(0, min(bins - 1, int((box[2] / max(1, screen_w)) * bins)))
        weight = max(0.3, float(item.get("conf") or 0.0)) * max(20.0, box_width(box))
        for idx in range(x1, x2 + 1):
            scores[idx] += weight
    peak = max(range(len(scores)), key=lambda idx: scores[idx])
    threshold = scores[peak] * 0.35
    left = peak
    right = peak
    while left > 0 and scores[left - 1] >= threshold:
        left -= 1
    while right < len(scores) - 1 and scores[right + 1] >= threshold:
        right += 1
    bin_w = max(1.0, screen_w / float(bins))
    return [int(round(left * bin_w)), 0, int(round((right + 1) * bin_w)), 0]


def _candidate_type(item: Dict[str, Any], screen_h: int) -> str:
    text = item.get("text") or ""
    box = item.get("bbox") or [0, 0, 0, 0]
    if header_like(text) and box[3] <= int(screen_h * 0.22):
        return "header"
    if question_like(text):
        return "question"
    if next_like(text):
        return "next"
    if item.get("has_frame"):
        return "framed_control"
    return "option"


def _filter_to_content(items: Sequence[Dict[str, Any]], content_column: Sequence[int], screen_h: int) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    col_x1, _, col_x2, _ = content_column
    for item in items:
        box = item["bbox"]
        cx, _ = box_center(box)
        item_type = _candidate_type(item, screen_h)
        item["candidate_type"] = item_type
        if item_type == "header":
            continue
        if cx < (col_x1 - 40) or cx > (col_x2 + 40):
            continue
        filtered.append(item)
    return filtered


def _prompt_candidates(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompts = [item for item in items if item.get("candidate_type") == "question"]
    if prompts:
        return prompts
    fallback = []
    for item in items:
        text = item.get("text") or ""
        if len(text) >= 12 and float(item.get("conf") or 0.0) >= 0.55:
            fallback.append(item)
    return fallback


def _build_option_objects(items: Sequence[Dict[str, Any]], question_box: Sequence[int], next_y: int) -> List[Dict[str, Any]]:
    options: List[Dict[str, Any]] = []
    seen_text = set()
    for item in items:
        box = item["bbox"]
        if box[1] <= question_box[3]:
            continue
        if next_y > 0 and box[3] >= next_y:
            continue
        if item.get("candidate_type") in {"question", "header"}:
            continue
        if item.get("has_frame"):
            continue
        if float(item.get("conf") or 0.0) < 0.35:
            continue
        text = clean_option_text(item.get("text") or "")
        norm = normalize_match_text(text)
        if not norm or norm in seen_text:
            continue
        seen_text.add(norm)
        options.append(
            {
                "id": item["id"],
                "text": text,
                "norm_text": norm,
                "bbox": [int(v) for v in box],
                "confidence": float(item.get("conf") or 0.0),
            }
        )
    options.sort(key=lambda opt: (opt["bbox"][1], opt["bbox"][0]))
    return options


def _guess_next_bbox(section_items: Sequence[Dict[str, Any]], section_end_y: int) -> Optional[List[int]]:
    text_matches = [
        item["bbox"]
        for item in section_items
        if item.get("candidate_type") == "next" and item["bbox"][1] >= int(section_end_y * 0.5)
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
        if not text or len(text) > 14:
            continue
        buttonish.append(item)
    if buttonish:
        buttonish.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        return [int(v) for v in buttonish[0]["bbox"]]
    return None


def _build_question_blocks(items: Sequence[Dict[str, Any]], screen_h: int, content_column: Sequence[int]) -> List[Dict[str, Any]]:
    prompts = _prompt_candidates(items)
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
            if item.get("has_frame") and item["bbox"][1] >= box[3]
        ]
        next_bbox = _guess_next_bbox(section_items, next_prompt_y)
        next_y = next_bbox[1] if next_bbox else next_prompt_y
        options = _build_option_objects(section_items, box, next_y)
        control_kind = "choice"
        select_bbox = None
        input_bbox = None
        if framed:
            frame_box = framed[0].get("dropdown_box") or framed[0]["bbox"]
            select_bbox = [int(v) for v in frame_box]
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
    return questions


def parse_screen_quiz_state(
    *,
    region_payload: Optional[Dict[str, Any]],
    summary_data: Optional[Dict[str, Any]] = None,
    page_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = region_payload or {}
    screen_w, screen_h = _image_size(payload)
    items = _prepare_items(payload)
    content_column = _content_column(items, screen_w)
    filtered = _filter_to_content(items, content_column, screen_h)
    questions = _build_question_blocks(filtered, screen_h, content_column)
    active = questions[0] if questions else None
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
    confidence["merged"] = confidence["screen"]
    return {
        "page_signature": page_signature,
        "screen_signature": screen_signature,
        "questions": questions,
        "active_question_id": active.get("id") if active else None,
        "active_question_signature": active.get("question_signature") if active else None,
        "question_text": active.get("prompt_text") if active else "",
        "options": (active.get("options") or []) if active else [],
        "control_kind": active.get("control_kind") if active else "unknown",
        "input_bbox": active.get("input_bbox") if active else None,
        "select_bbox": active.get("select_bbox") if active else None,
        "next_bbox": active.get("next_bbox") if active else None,
        "scroll_needed": bool(active.get("scroll_needed")) if active else False,
        "content_column": [int(content_column[0]), 0, int(content_column[2]), int(screen_h)],
        "confidence": confidence,
        "image": payload.get("image"),
        "summary_source": (summary_data or {}).get("_source"),
        "parser_debug": {
            "screen_size": {"width": screen_w, "height": screen_h},
            "items_considered": len(items),
            "items_filtered": len(filtered),
            "content_column": [int(content_column[0]), 0, int(content_column[2]), int(screen_h)],
        },
    }
