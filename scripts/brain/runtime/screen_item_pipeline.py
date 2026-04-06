from __future__ import annotations

from statistics import median
from typing import Any, Callable, Dict, List, Optional, Sequence

from .quiz_utils import box_center, box_iou, box_width, clean_option_text, normalize_match_text, normalize_ocr_text, text_similarity


def dedupe_items(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def prepare_items(
    payload: Dict[str, Any],
    *,
    extract_box_fn: Callable[[Dict[str, Any]], Optional[List[int]]],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for idx, raw in enumerate(payload.get("results") or []):
        if not isinstance(raw, dict):
            continue
        box = extract_box_fn(raw)
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
    return dedupe_items(items)


def option_horiz_overlap_ratio(box: Sequence[int], content_column: Sequence[int]) -> float:
    if len(box) != 4 or len(content_column) != 4:
        return 0.0
    x1 = int(box[0])
    x2 = int(box[2])
    c1 = int(content_column[0])
    c2 = int(content_column[2])
    width = max(1, x2 - x1)
    inter = max(0, min(x2, c2) - max(x1, c1))
    return float(inter) / float(width)


def prune_option_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    content_column: Sequence[int],
    question_bbox: Optional[Sequence[int]],
    bbox4_fn: Callable[[Any], Optional[List[int]]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    qbox = bbox4_fn(list(question_bbox) if isinstance(question_bbox, list) else question_bbox) if question_bbox else None
    col_x1 = int(content_column[0]) if len(content_column) == 4 else 0
    col_x2 = int(content_column[2]) if len(content_column) == 4 else 99999
    for row in candidates or []:
        if not isinstance(row, dict):
            continue
        box = bbox4_fn(row.get("bbox"))
        if box is None:
            continue
        if qbox is not None and int(box[1]) <= int(qbox[3]):
            continue
        overlap = option_horiz_overlap_ratio(box, content_column)
        cx, _ = box_center(box)
        if overlap < 0.18 and not (col_x1 - 28 <= int(cx) <= col_x2 + 28):
            continue
        text = str(row.get("text") or "")
        norm = normalize_match_text(text)
        if not norm:
            continue
        if len(norm) == 1:
            if not (norm.isdigit() or norm in {"a", "b", "c", "d"}):
                continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(dict(row))
    return merge_multi_column_options(out)


def content_column(items: Sequence[Dict[str, Any]], screen_w: int) -> List[int]:
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


def candidate_type(
    item: Dict[str, Any],
    *,
    screen_h: int,
    page_header_like_fn: Callable[[str], bool],
    header_like_fn: Callable[[str], bool],
    looks_like_instruction_fn: Callable[[str], bool],
    next_like_fn: Callable[[str], bool],
) -> str:
    text = item.get("text") or ""
    box = item.get("bbox") or [0, 0, 0, 0]
    if (page_header_like_fn(text) or header_like_fn(text)) and box[3] <= int(screen_h * 0.26):
        return "page_header"
    if looks_like_instruction_fn(text):
        return "question_prompt"
    if next_like_fn(text):
        return "next_button"
    if item.get("has_frame"):
        return "dropdown_trigger"
    return "answer_option"


def filter_to_content(
    items: Sequence[Dict[str, Any]],
    *,
    content_column_value: Sequence[int],
    screen_h: int,
    candidate_type_fn: Callable[[Dict[str, Any], int], str],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    col_x1, _, col_x2, _ = content_column_value
    for item in items:
        box = item["bbox"]
        cx, _ = box_center(box)
        item_type = candidate_type_fn(item, screen_h)
        item["candidate_type"] = item_type
        if item_type == "page_header":
            continue
        if cx < (col_x1 - 40) or cx > (col_x2 + 40):
            continue
        filtered.append(item)
    return filtered


def prompt_candidates(
    items: Sequence[Dict[str, Any]],
    *,
    input_placeholder_like_fn: Callable[[str], bool],
    page_header_like_fn: Callable[[str], bool],
    looks_like_instruction_fn: Callable[[str], bool],
    looks_like_option_text_fn: Callable[[str], bool],
    next_like_fn: Callable[[str], bool],
) -> List[Dict[str, Any]]:
    role_prompts = [
        item
        for item in items
        if str(item.get("role_pred") or "") == "question" and float(item.get("role_conf") or 0.0) >= 0.42
        and (not input_placeholder_like_fn(item.get("text") or ""))
        and (not page_header_like_fn(item.get("text") or ""))
    ]
    if role_prompts:
        role_prompts.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        return role_prompts
    prompts = [
        item
        for item in items
        if item.get("candidate_type") == "question_prompt"
        and (not input_placeholder_like_fn(item.get("text") or ""))
    ]
    if prompts:
        return prompts
    hint_prompts = [
        item
        for item in items
        if looks_like_instruction_fn(item.get("text") or "")
        and (not input_placeholder_like_fn(item.get("text") or ""))
    ]
    if hint_prompts:
        return hint_prompts
    fallback = []
    for item in items:
        text = item.get("text") or ""
        if (
            len(text) >= 12
            and float(item.get("conf") or 0.0) >= 0.55
            and (not page_header_like_fn(text))
            and (not next_like_fn(text))
            and (not looks_like_option_text_fn(text))
        ):
            fallback.append(item)
    return fallback


def build_option_objects(
    items: Sequence[Dict[str, Any]],
    *,
    question_box: Sequence[int],
    next_y: int,
    looks_like_instruction_fn: Callable[[str], bool],
    page_header_like_fn: Callable[[str], bool],
    next_like_fn: Callable[[str], bool],
) -> List[Dict[str, Any]]:
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
        role_pred = str(item.get("role_pred") or "")
        role_conf = float(item.get("role_conf") or 0.0)
        if role_pred in {"question", "next", "noise"} and role_conf >= 0.40:
            continue
        if any(item.get("role_probs", {}).get(k, 0.0) > 0.55 for k in ("question", "next", "noise")):
            continue
        if item.get("has_frame"):
            continue
        if float(item.get("conf") or 0.0) < 0.35:
            continue
        text = clean_option_text(item.get("text") or "")
        if looks_like_instruction_fn(text):
            continue
        norm = normalize_match_text(text)
        if not norm or norm in seen_text:
            continue
        if prompt_norm and text_similarity(norm, prompt_norm) >= 0.86:
            continue
        if page_header_like_fn(text) or next_like_fn(text):
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
    return merge_multi_column_options(options)


def merge_multi_column_options(options: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    prepared: List[Dict[str, Any]] = []
    for option in options or []:
        if not isinstance(option, dict):
            continue
        box = option.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            continue
        cy = int((int(box[1]) + int(box[3])) / 2)
        prepared.append({**option, "_row_cy": cy})
    if not prepared:
        return []
    heights = [max(1, int(opt["bbox"][3]) - int(opt["bbox"][1])) for opt in prepared]
    row_tolerance = max(10, int(round(float(median(heights)) * 0.65)))
    prepared.sort(key=lambda opt: (int(opt["_row_cy"]), int(opt["bbox"][0])))
    for option in prepared:
        row = None
        for candidate in rows:
            if abs(int(option["_row_cy"]) - int(candidate["center_y"])) <= row_tolerance:
                row = candidate
                break
        if row is None:
            row = {"center_y": int(option["_row_cy"]), "items": []}
            rows.append(row)
        row["items"].append(option)
        centers = [int(item["_row_cy"]) for item in row["items"]]
        row["center_y"] = int(round(sum(centers) / max(1, len(centers))))
    rows.sort(key=lambda row: int(row["center_y"]))
    merged: List[Dict[str, Any]] = []
    for row in rows:
        items = sorted(row["items"], key=lambda opt: (int(opt["bbox"][0]), int(opt["_row_cy"])))
        for item in items:
            out = dict(item)
            out.pop("_row_cy", None)
            merged.append(out)
    return merged
