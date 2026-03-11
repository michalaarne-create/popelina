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


_PAGE_HEADER_TOKENS = (
    "home",
    "test quiz server",
    "radio +",
    "checkbox",
    "dropdown",
    "input + next",
    "jednokrotna odpowied",
    "wielokrotna odpowied",
)
_MULTI_HINT_TOKENS = ("zaznacz", "wybierz wszystkie", "wielokrot")
_SCROLL_HINT_TOKENS = ("scroll", "przew", "właściwą opcję")
_TRIPLE_HINT_TOKENS = ("(1/3)", "(2/3)", "(3/3)", "1/3", "2/3", "3/3")
_MIX_HINT_TOKENS = ("(mix)", "mix)")


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


def _page_header_like(text: str) -> bool:
    norm = normalize_match_text(text)
    return bool(norm) and any(token in norm for token in _PAGE_HEADER_TOKENS)


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
    if (_page_header_like(text) or header_like(text)) and box[3] <= int(screen_h * 0.26):
        return "page_header"
    if question_like(text):
        return "question_prompt"
    if next_like(text):
        return "next_button"
    if item.get("has_frame"):
        return "dropdown_trigger"
    return "answer_option"


def _filter_to_content(items: Sequence[Dict[str, Any]], content_column: Sequence[int], screen_h: int) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    col_x1, _, col_x2, _ = content_column
    for item in items:
        box = item["bbox"]
        cx, _ = box_center(box)
        item_type = _candidate_type(item, screen_h)
        item["candidate_type"] = item_type
        if item_type == "page_header":
            continue
        if cx < (col_x1 - 40) or cx > (col_x2 + 40):
            continue
        filtered.append(item)
    return filtered


def _prompt_candidates(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompts = [item for item in items if item.get("candidate_type") == "question_prompt"]
    if prompts:
        return prompts
    fallback = []
    for item in items:
        text = item.get("text") or ""
        if len(text) >= 12 and float(item.get("conf") or 0.0) >= 0.55 and not _page_header_like(text) and not next_like(text):
            fallback.append(item)
    return fallback


def _build_option_objects(items: Sequence[Dict[str, Any]], question_box: Sequence[int], next_y: int) -> List[Dict[str, Any]]:
    options: List[Dict[str, Any]] = []
    seen_text = set()
    prompt_norm = ""
    for item in items:
        if list(item.get("bbox") or []) == list(question_box):
            prompt_norm = normalize_match_text(item.get("text") or "")
            break
    for item in items:
        box = item["bbox"]
        if box[1] <= question_box[3]:
            continue
        if next_y > 0 and box[3] >= next_y:
            continue
        if item.get("candidate_type") in {"question_prompt", "page_header", "next_button"}:
            continue
        if item.get("has_frame"):
            continue
        if float(item.get("conf") or 0.0) < 0.35:
            continue
        text = clean_option_text(item.get("text") or "")
        norm = normalize_match_text(text)
        if not norm or norm in seen_text:
            continue
        if prompt_norm and text_similarity(norm, prompt_norm) >= 0.86:
            continue
        if _page_header_like(text) or next_like(text):
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


def _summary_answer_candidates(summary_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(summary_data, dict):
        return []
    out: List[Dict[str, Any]] = []
    seen = set()
    for raw in summary_data.get("answer_candidate_boxes") or []:
        if not isinstance(raw, dict):
            continue
        box = _bbox4(raw.get("bbox"))
        text = clean_option_text(raw.get("text") or "")
        norm = normalize_match_text(text)
        if box is None or not norm or norm in seen:
            continue
        if float(raw.get("score") or 0.0) < 0.35:
            continue
        if _page_header_like(text) or next_like(text) or question_like(text):
            continue
        seen.add(norm)
        out.append(
            {
                "id": str(raw.get("id") or f"summary_answer_{len(out)}"),
                "text": text,
                "norm_text": norm,
                "bbox": box,
                "confidence": float(raw.get("score") or 0.0),
                "source": "summary_answer_candidates",
            }
        )
    out.sort(key=lambda opt: (opt["bbox"][1], opt["bbox"][0]))
    return out


def _summary_first_bbox(summary_data: Optional[Dict[str, Any]], *labels: str) -> Optional[List[int]]:
    if not isinstance(summary_data, dict):
        return None
    top = summary_data.get("top_labels") or {}
    for label in labels:
        entry = top.get(label)
        if isinstance(entry, dict):
            text = str(entry.get("text") or "")
            if label.startswith("next") and (_page_header_like(text) or question_like(text)):
                continue
            if label == "dropdown" and _page_header_like(text):
                continue
            box = _bbox4(entry.get("bbox"))
            if box is not None:
                if label.startswith("next") and box[1] < 140:
                    continue
                return box
    return None


def _guess_next_bbox(section_items: Sequence[Dict[str, Any]], section_end_y: int) -> Optional[List[int]]:
    text_matches = [
        item["bbox"]
        for item in section_items
        if item.get("candidate_type") == "next_button" and item["bbox"][1] >= int(section_end_y * 0.5)
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
        if not text or len(text) > 14 or _page_header_like(text):
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
            if item.get("candidate_type") == "dropdown_trigger" and item["bbox"][1] >= box[3]
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

    def _ret(qtype: str, conf: float) -> Dict[str, Any]:
        op = "text" if qtype == "text" else ("dropdown" if qtype in {"dropdown", "dropdown_scroll"} else "choice")
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
    summary_answer_candidates = _summary_answer_candidates(summary_data)
    if active:
        if (not active.get("next_bbox")) and isinstance(summary_data, dict):
            active["next_bbox"] = _summary_first_bbox(summary_data, "next_active", "next_inactive")
        if (not active.get("select_bbox")) and active.get("control_kind") == "dropdown":
            active["select_bbox"] = _summary_first_bbox(summary_data, "dropdown")
        if active.get("control_kind") == "text" and not active.get("input_bbox"):
            content_box = [int(content_column[0]), 0, int(content_column[2]), int(screen_h)]
            active["input_bbox"] = [
                int(content_box[0]),
                int(max(active["bbox"][3] + 12, screen_h * 0.28)),
                int(content_box[2]),
                int(min(screen_h - 10, (active.get("next_bbox") or [0, screen_h, 0, screen_h])[1] - 12)),
            ]
            if active["input_bbox"][3] <= active["input_bbox"][1]:
                active["input_bbox"][3] = min(screen_h - 5, active["input_bbox"][1] + 44)
        if summary_answer_candidates:
            existing = {opt.get("norm_text") for opt in (active.get("options") or []) if isinstance(opt, dict)}
            for candidate in summary_answer_candidates:
                if candidate.get("norm_text") in existing:
                    continue
                active.setdefault("options", []).append(candidate)
            active["options"].sort(key=lambda opt: (opt["bbox"][1], opt["bbox"][0]))
    type_detection = _detect_screen_quiz_type(active)
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
        "active_question_id": active.get("id") if active else None,
        "active_question_signature": active.get("question_signature") if active else None,
        "question_text": active.get("prompt_text") if active else "",
        "options": (active.get("options") or []) if active else [],
        "control_kind": active.get("control_kind") if active else "unknown",
        "detected_quiz_type": type_detection.get("detected_quiz_type"),
        "detected_operational_type": type_detection.get("detected_operational_type"),
        "type_confidence": type_detection.get("type_confidence"),
        "type_source": type_detection.get("type_source"),
        "type_signals": type_detection.get("type_signals"),
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
            "content_column": [int(content_column[0]), 0, int(content_column[2]), int(screen_h)],
            "parse_quality": parse_quality,
        },
    }
