from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence


_TOKEN_WEIGHTS = {
    "submit": 3.0,
    "answer": 2.5,
    "question": 2.5,
    "next": 2.0,
    "prev": 1.5,
    "back": 1.5,
    "modal": 2.0,
    "dialog": 2.0,
    "error": 2.5,
    "warning": 2.0,
    "banner": 1.5,
    "dropdown": 2.5,
    "select": 2.0,
    "radio": 2.5,
    "checkbox": 2.5,
    "textbox": 2.0,
    "input": 2.0,
    "save": 2.0,
    "review": 2.0,
}


def _tokenize(value: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", str(value or "").strip().lower()) if token]


def _collect_tokens(values: Iterable[Any]) -> List[str]:
    tokens: List[str] = []
    for value in values:
        tokens.extend(_tokenize(str(value)))
    return tokens


def probe_css_semantics(
    *,
    class_names: Sequence[str] | None = None,
    element_ids: Sequence[str] | None = None,
    text_values: Sequence[str] | None = None,
) -> Dict[str, Any]:
    class_tokens = _collect_tokens(class_names or [])
    id_tokens = _collect_tokens(element_ids or [])
    text_tokens = _collect_tokens(text_values or [])
    all_tokens = class_tokens + id_tokens + text_tokens

    scores: Dict[str, float] = {}
    evidence: Dict[str, List[str]] = {}
    for semantic, weight in _TOKEN_WEIGHTS.items():
        matched = [token for token in all_tokens if semantic in token or token == semantic]
        if matched:
            scores[semantic] = scores.get(semantic, 0.0) + weight * len(matched)
            evidence[semantic] = matched[:5]

    if scores:
        best_semantic = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
        confidence = min(1.0, scores[best_semantic] / 5.0)
        reason = "semantic_css_hint"
    else:
        best_semantic = "unknown"
        confidence = 0.0
        reason = "no_semantic_css_hint"

    return {
        "best_semantic": best_semantic,
        "confidence": round(float(confidence), 4),
        "reason": reason,
        "scores": {key: round(float(value), 4) for key, value in scores.items()},
        "evidence": evidence,
        "tokens": {
            "class_tokens": class_tokens,
            "id_tokens": id_tokens,
            "text_tokens": text_tokens,
        },
    }
