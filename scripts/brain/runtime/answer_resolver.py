from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .quiz_types import ResolvedQuizAnswer
from .quiz_utils import (
    best_text_match,
    extract_arithmetic_answer,
    normalize_match_text,
    normalize_question_text,
    normalized_options_texts,
    quoted_answer,
    signature_for_question,
    text_similarity,
)

_CHOICE_FAMILY = {"choice", "single", "multi", "triple", "mixed"}


def _question_core(text: str) -> str:
    raw = normalize_match_text(text or "")
    if not raw:
        return ""
    raw = re.sub(r"^\(?\d+\s*/\s*\d+\)?\s*", "", raw).strip()
    raw = re.sub(r"^\(?mix\)?\s*", "", raw).strip()
    return raw


def _compatible_question_types(qtype: str) -> List[str]:
    t = str(qtype or "").strip().lower() or "single"
    if t in _CHOICE_FAMILY:
        return ["single", "multi", "triple", "mixed", "choice"]
    if t == "dropdown":
        return ["dropdown", "dropdown_scroll"]
    if t == "dropdown_scroll":
        return ["dropdown_scroll", "dropdown"]
    return [t]


def _is_meaningful_screen_option(text: str) -> bool:
    norm = normalize_match_text(text or "")
    if not norm:
        return False
    return any(ch.isalnum() for ch in norm)


def _load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"items": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {"items": {}}
    if not isinstance(payload, dict):
        return {"items": {}}
    payload.setdefault("items", {})
    return payload


def _build_cache_indexes(payload: Dict[str, Any]) -> Dict[str, Any]:
    items = payload.get("items") or {}
    by_qid: Dict[str, Dict[str, Any]] = {}
    by_sig: Dict[str, Dict[str, Any]] = {}
    normalized_items: List[Dict[str, Any]] = []
    for qid, raw in items.items():
        if not isinstance(raw, dict):
            continue
        entry = dict(raw)
        entry["question_key"] = qid
        entry["question_text_norm"] = normalize_match_text(entry.get("question_text") or "")
        entry["question_core_norm"] = _question_core(entry.get("question_text") or "")
        entry["options_list"] = normalized_options_texts(entry.get("options_text"))
        qtype = str(entry.get("question_type") or "single")
        entry["signature"] = signature_for_question(entry.get("question_text") or "", entry["options_list"], qtype)
        by_qid[qid] = entry
        by_sig[entry["signature"]] = entry
        normalized_items.append(entry)
    return {"by_qid": by_qid, "by_sig": by_sig, "items": normalized_items}


def _resolved_from_entry(entry: Dict[str, Any], *, source: str, confidence: float) -> ResolvedQuizAnswer:
    correct_answers: List[str] = []
    if entry.get("question_type") == "text":
        if entry.get("text_answer"):
            correct_answers = [str(entry.get("text_answer"))]
        elif entry.get("correct_answer"):
            correct_answers = [str(entry.get("correct_answer"))]
    elif entry.get("correct_answer"):
        correct_answers = [str(entry.get("correct_answer"))]
    option_indexes: List[int] = []
    options_list = normalized_options_texts(entry.get("options_text"))
    if correct_answers:
        for answer in correct_answers:
            _, score, idx = best_text_match(answer, options_list)
            if idx >= 0 and score >= 0.8 and idx not in option_indexes:
                option_indexes.append(idx)
    if not correct_answers and isinstance(entry.get("selected_options"), list) and isinstance(entry.get("options_text"), dict):
        letters = list(sorted((entry.get("options_text") or {}).keys()))
        for letter in entry.get("selected_options") or []:
            if letter in letters:
                idx = letters.index(letter)
                option_indexes.append(idx)
                correct_answers.append(str((entry.get("options_text") or {}).get(letter) or ""))
    return ResolvedQuizAnswer(
        matched=True,
        question_key=str(entry.get("question_key") or ""),
        question_type=str(entry.get("question_type") or "single"),
        correct_answers=correct_answers,
        option_indexes=option_indexes,
        source=source,
        confidence=confidence,
        fingerprint=str(entry.get("signature") or ""),
        question_text=str(entry.get("question_text") or ""),
        normalized_question_text=str(entry.get("question_text_norm") or ""),
        cache_item=entry,
    )


def resolve_answer(
    *,
    cache_path: Path,
    screen_state: Dict[str, Any],
    controls_data: Optional[Dict[str, Any]] = None,
) -> ResolvedQuizAnswer:
    cache = _build_cache_indexes(_load_cache(cache_path))
    by_qid = cache["by_qid"]
    by_sig = cache["by_sig"]
    items = cache["items"]
    active_question = str(screen_state.get("question_text") or "")
    question_type = str(
        screen_state.get("detected_quiz_type")
        or screen_state.get("control_kind")
        or "single"
    )
    options = [str((opt or {}).get("text") or "") for opt in (screen_state.get("options") or [])]
    screen_opts = [normalize_match_text(opt) for opt in options if _is_meaningful_screen_option(opt)]
    screen_has_select = bool(screen_state.get("select_bbox"))
    screen_has_next = bool(screen_state.get("next_bbox"))
    screen_scroll_needed = bool(screen_state.get("scroll_needed"))

    signature_hits = []
    for qt in _compatible_question_types(question_type):
        signature = signature_for_question(active_question, options, qt)
        if signature in by_sig:
            signature_hits.append(by_sig[signature])
    if signature_hits:
        best = signature_hits[0]
        return _resolved_from_entry(best, source="signature", confidence=0.98)

    norm_question = normalize_match_text(active_question)
    norm_question_core = _question_core(active_question)
    best_entry = None
    best_score = 0.0
    best_question_score = 0.0
    for entry in items:
        q_score_full = text_similarity(norm_question, entry.get("question_text_norm") or "")
        q_score_core = text_similarity(norm_question_core, entry.get("question_core_norm") or "")
        q_score = max(q_score_full, q_score_core)
        if q_score < 0.72:
            continue
        if q_score > best_question_score:
            best_question_score = q_score
        entry_qtype = str(entry.get("question_type") or "")
        if entry_qtype == question_type:
            type_bonus = 0.08
        elif entry_qtype in _compatible_question_types(question_type):
            type_bonus = 0.05
        else:
            type_bonus = 0.0
        entry_opts = [normalize_match_text(opt) for opt in entry.get("options_list") or [] if normalize_match_text(opt)]
        overlap = 0.0
        if screen_opts and entry_opts:
            screen_set = set(screen_opts)
            entry_set = set(entry_opts)
            overlap = len(screen_set & entry_set) / float(max(1, len(entry_set)))
        layout_bonus = 0.0
        if entry_qtype in {"dropdown", "dropdown_scroll"}:
            if screen_has_select:
                layout_bonus += 0.08
            elif (not screen_opts) and screen_has_next:
                layout_bonus += 0.10
            elif len(screen_opts) >= 2:
                layout_bonus -= 0.22
        elif entry_qtype in _CHOICE_FAMILY:
            if len(screen_opts) >= 2:
                layout_bonus += 0.08
            elif question_type == "text" and (not screen_has_select) and screen_has_next:
                layout_bonus += 0.05
        if entry_qtype == "triple":
            if screen_scroll_needed:
                layout_bonus += 0.04
            else:
                layout_bonus -= 0.08
                if (not screen_opts) and screen_has_next:
                    layout_bonus -= 0.14
            if q_score_core >= 0.96:
                layout_bonus += 0.03
        score = q_score * 0.75 + overlap * 0.20 + type_bonus + layout_bonus
        if score > best_score:
            best_score = score
            best_entry = entry
    if best_entry and best_score >= 0.76:
        return _resolved_from_entry(best_entry, source="fuzzy", confidence=min(0.95, best_score))
    if best_entry and not screen_opts:
        entry_qtype = str(best_entry.get("question_type") or "single").strip().lower()
        min_question_only_score = 0.90
        if entry_qtype in {"dropdown", "dropdown_scroll"}:
            min_question_only_score = 0.84
        if (entry_qtype in _CHOICE_FAMILY or entry_qtype in {"dropdown", "dropdown_scroll"}) and best_question_score >= min_question_only_score:
            return _resolved_from_entry(
                best_entry,
                source="question_only_cache",
                confidence=min(0.92, best_question_score),
            )

    qid = None
    if isinstance(controls_data, dict):
        meta = controls_data.get("meta") or {}
        qid = meta.get("qid")
        if not qid:
            blocks = controls_data.get("question_blocks") or []
            if blocks and isinstance(blocks[0], dict):
                qid = blocks[0].get("qid")
    if qid and str(qid) in by_qid:
        return _resolved_from_entry(by_qid[str(qid)], source="qid_fallback", confidence=0.9)

    if question_type == "text" or not options:
        quoted = quoted_answer(active_question)
        if quoted:
            return ResolvedQuizAnswer(
                matched=True,
                question_key=None,
                question_type="text",
                correct_answers=[quoted],
                source="heuristic_quoted",
                confidence=0.7,
                question_text=normalize_question_text(active_question),
                normalized_question_text=norm_question,
            )
        arithmetic = extract_arithmetic_answer(active_question)
        if arithmetic is not None:
            return ResolvedQuizAnswer(
                matched=True,
                question_key=None,
                question_type="text",
                correct_answers=[arithmetic],
                source="heuristic_arithmetic",
                confidence=0.72,
                question_text=normalize_question_text(active_question),
                normalized_question_text=norm_question,
            )

    return ResolvedQuizAnswer(
        matched=False,
        question_key=None,
        question_type=question_type,
        correct_answers=[],
        source="unresolved",
        confidence=0.0,
        question_text=normalize_question_text(active_question),
        normalized_question_text=norm_question,
    )
