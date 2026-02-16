from __future__ import annotations

import hashlib
import re
from typing import List

from .types import Element, QuestionCluster, QuestionOption


def infer_question_type(question: Element) -> str:
    score_single = float(question.scores_fused.get("answer_single", 0.0))
    score_multi = float(question.scores_fused.get("answer_multi", 0.0))
    return "multi" if score_multi > score_single else "single"


def infer_ui_mode(question: Element, options: List[QuestionOption]) -> str:
    dropdown_opts = [opt for opt in options if (opt.metadata.get("ui_hints") or {}).get("dropdown_like")]
    q_hint = (question.metadata.get("ui_hints") or {}).get("dropdown_like")
    if dropdown_opts or q_hint:
        return "dropdown"
    return "standard"


def make_canonical_key(question: Element, options: List[QuestionOption], q_type: str) -> str:
    q_text = (question.text_dom or question.text_ocr or "").strip().lower()
    opt_texts = [opt.text.strip().lower() for opt in options]
    payload = f"{q_type}\n{q_text}\n" + "\n".join(sorted(opt_texts))
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def tag_question(question: Element, cfg) -> List[str]:
    text = (question.text_dom or question.text_ocr or "").lower()
    tags: List[str] = []
    for tag, keywords in cfg.topic_keywords.items():
        if any(re.search(rf"\b{kw}\b", text) for kw in keywords):
            tags.append(tag)
    return tags


def deduplicate_clusters(clusters: List[QuestionCluster]) -> List[QuestionCluster]:
    seen = set()
    unique: List[QuestionCluster] = []
    for cl in clusters:
        key = cl.canonical_key or cl.id
        if key in seen:
            continue
        seen.add(key)
        unique.append(cl)
    return unique
