from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from .quiz_utils import clean_option_text, normalize_match_text, normalize_question_text, question_like


def summary_answer_candidates(
    summary_data: Optional[Dict[str, Any]],
    *,
    bbox4_fn: Callable[[Any], Optional[List[int]]],
    looks_like_instruction_fn: Callable[[str], bool],
    page_header_like_fn: Callable[[str], bool],
    next_like_fn: Callable[[str], bool],
) -> List[Dict[str, Any]]:
    if not isinstance(summary_data, dict):
        return []
    out: List[Dict[str, Any]] = []
    seen = set()
    for raw in summary_data.get("answer_candidate_boxes") or []:
        if not isinstance(raw, dict):
            continue
        box = bbox4_fn(raw.get("bbox"))
        text = clean_option_text(raw.get("text") or "")
        norm = normalize_match_text(text)
        if box is None or not norm or norm in seen:
            continue
        score = float(raw.get("score") or 0.0)
        marker_kind = str(raw.get("marker_kind") or "none")
        if score < 0.35 and marker_kind not in {"radio", "checkbox"}:
            continue
        if looks_like_instruction_fn(text):
            continue
        if page_header_like_fn(text) or next_like_fn(text) or question_like(text):
            continue
        seen.add(norm)
        out.append(
            {
                "id": str(raw.get("id") or f"summary_answer_{len(out)}"),
                "text": text,
                "norm_text": norm,
                "bbox": box,
                "confidence": score,
                "source": "summary_answer_candidates",
            }
        )
    out.sort(key=lambda opt: (opt["bbox"][1], opt["bbox"][0]))
    return out


def summary_question_candidate(
    summary_data: Optional[Dict[str, Any]],
    *,
    bbox4_fn: Callable[[Any], Optional[List[int]]],
    looks_like_instruction_fn: Callable[[str], bool],
) -> Optional[Dict[str, Any]]:
    if not isinstance(summary_data, dict):
        return None
    row = summary_data.get("question_candidate")
    if isinstance(row, dict):
        box = bbox4_fn(row.get("bbox"))
        text = normalize_question_text(str(row.get("text") or ""))
        if box is not None and text:
            return {
                "id": str(row.get("id") or "summary_question"),
                "text": text,
                "norm_text": normalize_match_text(text),
                "bbox": box,
                "score": float(row.get("score") or 0.0),
                "source": "summary_question_candidate",
            }
    top = summary_data.get("top_labels") if isinstance(summary_data.get("top_labels"), dict) else {}
    for label in ("answer_multi", "answer_single"):
        row = top.get(label)
        if not isinstance(row, dict):
            continue
        text = normalize_question_text(str(row.get("text") or ""))
        box = bbox4_fn(row.get("bbox"))
        if box is None or not text:
            continue
        if question_like(text) or looks_like_instruction_fn(text):
            return {
                "id": str(row.get("id") or "summary_top_question"),
                "text": text,
                "norm_text": normalize_match_text(text),
                "bbox": box,
                "score": float(row.get("score") or 0.0),
                "source": "summary_top_label",
            }
    return None


def rated_answer_candidates(
    rated_data: Optional[Dict[str, Any]],
    *,
    question_text: str,
    question_bbox: Optional[List[int]],
    bbox4_fn: Callable[[Any], Optional[List[int]]],
    looks_like_instruction_fn: Callable[[str], bool],
    page_header_like_fn: Callable[[str], bool],
    next_like_fn: Callable[[str], bool],
) -> List[Dict[str, Any]]:
    if not isinstance(rated_data, dict):
        return []
    out: List[Dict[str, Any]] = []
    seen = set()
    q_norm = normalize_match_text(question_text)
    for raw in rated_data.get("elements") or []:
        if not isinstance(raw, dict):
            continue
        box = bbox4_fn(raw.get("bbox"))
        text = clean_option_text(raw.get("text") or raw.get("box") or "")
        norm = normalize_match_text(text)
        if box is None or not norm or norm in seen:
            continue
        if q_norm and norm == q_norm:
            continue
        if len(text) > 80:
            continue
        if question_bbox is not None and int(box[1]) <= int(question_bbox[3]):
            continue
        if looks_like_instruction_fn(text):
            continue
        if page_header_like_fn(text) or next_like_fn(text) or question_like(text):
            continue
        marker = raw.get("marker") if isinstance(raw.get("marker"), dict) else {}
        scores = raw.get("scores") if isinstance(raw.get("scores"), dict) else {}
        score = max(float(scores.get("answer_multi") or 0.0), float(scores.get("answer_single") or 0.0))
        if score < 0.35 and str(marker.get("kind") or "none") not in {"radio", "checkbox"}:
            continue
        seen.add(norm)
        out.append(
            {
                "id": str(raw.get("id") or f"rated_answer_{len(out)}"),
                "text": text,
                "norm_text": norm,
                "bbox": box,
                "confidence": float(round(score, 4)),
                "source": "rated_answer_candidates",
            }
        )
    out.sort(key=lambda opt: (opt["bbox"][1], opt["bbox"][0]))
    return out


def summary_first_bbox(
    summary_data: Optional[Dict[str, Any]],
    *labels: str,
    bbox4_fn: Callable[[Any], Optional[List[int]]],
    page_header_like_fn: Callable[[str], bool],
) -> Optional[List[int]]:
    if not isinstance(summary_data, dict):
        return None
    top = summary_data.get("top_labels") or {}
    for label in labels:
        entry = top.get(label)
        if isinstance(entry, dict):
            text = str(entry.get("text") or "")
            if label.startswith("next") and (page_header_like_fn(text) or question_like(text)):
                continue
            if label == "dropdown" and page_header_like_fn(text):
                continue
            box = bbox4_fn(entry.get("bbox"))
            if box is not None:
                if label.startswith("next") and box[1] < 140:
                    continue
                return box
    return None


def summary_dropdown_candidates(
    summary_data: Optional[Dict[str, Any]],
    *,
    bbox4_fn: Callable[[Any], Optional[List[int]]],
    page_header_like_fn: Callable[[str], bool],
) -> List[Dict[str, Any]]:
    if not isinstance(summary_data, dict):
        return []
    out: List[Dict[str, Any]] = []
    for raw in summary_data.get("dropdown_candidate_boxes") or []:
        if not isinstance(raw, dict):
            continue
        box = bbox4_fn(raw.get("bbox"))
        text = str(raw.get("text") or "")
        if box is None or (not text) or page_header_like_fn(text):
            continue
        out.append(
            {
                "id": str(raw.get("id") or f"summary_dropdown_{len(out)}"),
                "text": text,
                "norm_text": normalize_match_text(text),
                "bbox": box,
                "score": float(raw.get("score") or 0.0),
            }
        )
    out.sort(key=lambda item: (-float(item.get("score") or 0.0), item["bbox"][1], item["bbox"][0]))
    return out
