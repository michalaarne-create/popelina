from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def collect_texts_from_payload(
    payload: Any,
    *,
    text_keys: Sequence[str] = ("text", "box", "prompt_text", "question_text"),
) -> List[str]:
    out: List[str] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in text_keys and isinstance(value, str):
                    txt = value.strip()
                    if txt:
                        out.append(txt)
                elif isinstance(value, (dict, list)):
                    _walk(value)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(payload)
    return out


def build_prompt_artifact_signals(
    *,
    region_payload: Optional[Dict[str, Any]],
    summary_data: Optional[Dict[str, Any]],
    rated_data: Optional[Dict[str, Any]],
    questions: Sequence[Dict[str, Any]],
    active: Optional[Dict[str, Any]],
    prompt_triple: Sequence[str],
    prompt_mix: Sequence[str],
    prompt_slider: Sequence[str],
    prompt_scroll: Sequence[str],
    prompt_multi: Sequence[str],
) -> Dict[str, Any]:
    texts: List[str] = []
    for q in questions or []:
        if not isinstance(q, dict):
            continue
        for key in ("prompt_text", "question_text"):
            raw = str(q.get(key) or "").strip()
            if raw:
                texts.append(raw)
    if isinstance(active, dict):
        for key in ("prompt_text", "question_text"):
            raw = str(active.get(key) or "").strip()
            if raw:
                texts.append(raw)
    if isinstance(summary_data, dict):
        texts.extend(collect_texts_from_payload(summary_data))
    if isinstance(rated_data, dict):
        texts.extend(collect_texts_from_payload({"elements": rated_data.get("elements") or []}))
    if isinstance(region_payload, dict):
        texts.extend(collect_texts_from_payload({"results": region_payload.get("results") or []}))

    normalized = [" ".join(str(t).strip().lower().split()) for t in texts if str(t).strip()]
    triple_tokens_found: List[str] = []
    for tok in prompt_triple:
        if any(tok in txt for txt in normalized):
            triple_tokens_found.append(tok)
    return {
        "triple_hint": bool(triple_tokens_found),
        "triple_token_count": int(len(set(triple_tokens_found))),
        "mix_hint": any("(mix" in txt or "mix " in txt or any(tok in txt for tok in prompt_mix) for txt in normalized),
        "slider_hint": any(tok in txt for txt in normalized for tok in prompt_slider),
        "scroll_hint": any(tok in txt for txt in normalized for tok in prompt_scroll),
        "multi_hint": any(tok in txt for txt in normalized for tok in prompt_multi),
    }


def has_structural_prompt_tokens(
    text: str,
    *,
    prompt_triple: Sequence[str],
    prompt_mix: Sequence[str],
) -> bool:
    raw = " ".join(str(text or "").strip().lower().split())
    if not raw:
        return False
    return any(tok in raw for tok in prompt_triple) or any(tok in raw for tok in prompt_mix)


def items_have_structural_tokens(
    items: Sequence[Dict[str, Any]],
    *,
    summary_data: Optional[Dict[str, Any]],
    prompt_triple: Sequence[str],
    prompt_mix: Sequence[str],
) -> bool:
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if has_structural_prompt_tokens(
            str(item.get("text") or ""),
            prompt_triple=prompt_triple,
            prompt_mix=prompt_mix,
        ):
            return True
    if isinstance(summary_data, dict):
        candidate = summary_data.get("question_candidate")
        if isinstance(candidate, dict) and has_structural_prompt_tokens(
            str(candidate.get("text") or ""),
            prompt_triple=prompt_triple,
            prompt_mix=prompt_mix,
        ):
            return True
        for key in ("answer_candidate_boxes", "dropdown_candidate_boxes"):
            for raw in summary_data.get(key) or []:
                if isinstance(raw, dict) and has_structural_prompt_tokens(
                    str(raw.get("text") or ""),
                    prompt_triple=prompt_triple,
                    prompt_mix=prompt_mix,
                ):
                    return True
    return False
