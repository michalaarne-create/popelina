from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from .screen_item_pipeline import merge_multi_column_options
from .quiz_utils import box_iou, box_width, normalize_match_text, normalize_question_text, sha1_text, text_similarity


def _option_texts_suggest_slider(options: Sequence[Dict[str, Any]]) -> bool:
    normalized = [normalize_match_text(str(opt.get("text") or "")) for opt in options if isinstance(opt, dict)]
    if not normalized:
        return False
    for text in normalized:
        if not text:
            continue
        if "%" in text or "/10" in text or "lat" in text:
            return True
        if any(tok in text for tok in ("slider", "progress", "intensity", "level", "temperature", "wiek")):
            return True
    return False


def _question_text_suggests_dropdown(prompt_norm: str) -> bool:
    if not prompt_norm:
        return False
    return any(tok in prompt_norm for tok in ("wybierz opcje", "choose one", "select option", "select one"))


def build_summary_backed_question(
    question_hint: Dict[str, Any],
    *,
    summary_data: Optional[Dict[str, Any]],
    summary_answer_candidates: Sequence[Dict[str, Any]],
    screen_w: int,
    screen_h: int,
    content_column: Sequence[int],
    scroll_hint_tokens: Sequence[str],
    summary_first_bbox_fn: Callable[..., Optional[List[int]]],
    prompt_prefers_text_fn: Callable[[str], bool],
    prompt_prefers_choice_fn: Callable[[str], bool],
    bbox4_fn: Callable[[Any], Optional[List[int]]],
    mode: str,
) -> Dict[str, Any]:
    select_bbox = None
    next_bbox = summary_first_bbox_fn(summary_data, "next_active", "next_inactive")
    prompt_text = normalize_question_text(str(question_hint.get("text") or ""))
    prompt_norm = normalize_match_text(prompt_text)
    options = list(summary_answer_candidates or [])
    control_kind = "dropdown" if select_bbox else ("text" if not options else "choice")
    if prompt_norm and prompt_prefers_text_fn(prompt_text):
        control_kind = "text"
    elif len(options) >= 3 and prompt_prefers_choice_fn(prompt_text):
        control_kind = "choice"
        select_bbox = None
    if options and any(tok in prompt_norm for tok in scroll_hint_tokens):
        control_kind = "dropdown"
    top = summary_data.get("top_labels") if isinstance(summary_data, dict) else {}
    dropdown_top = top.get("dropdown") if isinstance(top, dict) else None
    if not select_bbox and isinstance(dropdown_top, dict):
        dropdown_text = str(dropdown_top.get("text") or "")
        dropdown_norm = normalize_match_text(dropdown_text)
        dropdown_box = bbox4_fn(dropdown_top.get("bbox"))
        question_box = bbox4_fn(question_hint.get("bbox"))
        same_prompt = bool(prompt_norm and dropdown_norm and text_similarity(prompt_norm, dropdown_norm) >= 0.9)
        overlaps_prompt = bool(dropdown_box and question_box and box_iou(dropdown_box, question_box) >= 0.45)
        if same_prompt or overlaps_prompt:
            select_bbox = dropdown_box
            control_kind = "dropdown"
    if not select_bbox and options:
        option_norms = {normalize_match_text(str(opt.get("text") or "")) for opt in options if isinstance(opt, dict)}
        if any(tok in option_norms for tok in {"szukaj.", "szukaj", "expand"}):
            control_kind = "dropdown"
            if mode == "collapse":
                select_bbox = bbox4_fn(question_hint.get("bbox"))
    if control_kind == "text" and _question_text_suggests_dropdown(prompt_norm):
        control_kind = "dropdown"
        if mode == "collapse":
            select_bbox = bbox4_fn(question_hint.get("bbox"))
    if _option_texts_suggest_slider(options):
        control_kind = "slider"
        select_bbox = None
    question_signature = sha1_text(
        "\n".join(
            [prompt_norm] + [str(opt.get("norm_text") or "") for opt in options if isinstance(opt, dict)]
        )
        or f"__summary_{mode}__"
    )
    out = {
        "id": str(question_hint.get("id") or f"summary_{mode}"),
        "bbox": bbox4_fn(question_hint.get("bbox")) or [0, 0, max(1, int(screen_w * 0.7)), max(1, int(screen_h * 0.25))],
        "prompt_text": prompt_text,
        "prompt_norm": prompt_norm,
        "question_signature": question_signature,
        "control_kind": control_kind if mode == "collapse" else ("slider" if control_kind == "slider" else ("dropdown" if select_bbox else ("text" if not options else "unknown"))),
        "options": options,
        "next_bbox": next_bbox,
        "select_bbox": select_bbox,
        "input_bbox": None,
        "scroll_needed": bool(any(tok in prompt_norm for tok in scroll_hint_tokens)) if mode == "collapse" else False,
        "confidence": float(question_hint.get("score") or 0.0),
        "source": f"summary_{mode}" if mode == "collapse" else str(question_hint.get("source") or "summary"),
    }
    if mode != "collapse":
        out["prompt_conf"] = float(question_hint.get("score") or 0.0)
    if control_kind == "text" and mode == "collapse":
        out["input_bbox"] = [
            int(content_column[0]),
            int(max(out["bbox"][3] + 12, screen_h * 0.28)),
            int(content_column[2]),
            int(min(screen_h - 10, (next_bbox or [0, screen_h, 0, screen_h])[1] - 12)),
        ]
    return out


def enrich_active_question_from_summary(
    active: Optional[Dict[str, Any]],
    *,
    summary_data: Optional[Dict[str, Any]],
    summary_answer_candidates: Sequence[Dict[str, Any]],
    screen_h: int,
    content_column: Sequence[int],
    summary_first_bbox_fn: Callable[..., Optional[List[int]]],
    prune_option_candidates_fn: Callable[..., List[Dict[str, Any]]],
    bbox4_fn: Callable[[Any], Optional[List[int]]],
) -> None:
    if not isinstance(active, dict):
        return
    if (not active.get("next_bbox")) and isinstance(summary_data, dict):
        active["next_bbox"] = summary_first_bbox_fn(summary_data, "next_active", "next_inactive")
    if (not active.get("select_bbox")) and active.get("control_kind") == "dropdown":
        active["select_bbox"] = summary_first_bbox_fn(summary_data, "dropdown")
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
        active["options"] = merge_multi_column_options(active.get("options") or [])
    active["options"] = prune_option_candidates_fn(
        active.get("options") or [],
        content_column=content_column,
        question_bbox=active.get("bbox"),
    )
    if str(active.get("control_kind") or "") != "dropdown":
        return
    prompt_text = str(active.get("prompt_text") or "")
    prompt_box = bbox4_fn(active.get("bbox"))
    select_box = bbox4_fn(active.get("select_bbox"))
    opts = active.get("options") if isinstance(active.get("options"), list) else []
    if prompt_box is None or select_box is None:
        return
    prompt_overlap = box_iou(prompt_box, select_box)
    select_near_prompt = int(select_box[1]) <= int(prompt_box[3]) + 4
    very_wide_select = box_width(select_box) >= int(box_width(prompt_box) * 2.2)
    sparse_opts = len(opts) <= 1
    if ("?" in prompt_text) and sparse_opts and (select_near_prompt or prompt_overlap >= 0.28 or very_wide_select):
        active["control_kind"] = "choice"
        active["select_bbox"] = None
        if str(active.get("block_type") or "").startswith("dropdown"):
            active["block_type"] = "single"
