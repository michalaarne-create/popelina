from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

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
    raw = re.sub(r"\(?\d+\s*/\s*\d+\)?", " ", raw).strip()
    raw = re.sub(r"^\(?mix\)?\s*", "", raw).strip()
    return raw


def _question_match_variants(text: str) -> List[str]:
    variants: List[str] = []
    norm = normalize_match_text(text or "")
    if norm:
        variants.append(norm)
        inline_triple = re.match(r"^(.*?)(\(\d+\s*/\s*\d+\)\s*)(.+)$", norm)
        if inline_triple:
            prefix = normalize_match_text(inline_triple.group(1) or "")
            marker = normalize_match_text(inline_triple.group(2) or "")
            suffix = normalize_match_text(inline_triple.group(3) or "")
            if suffix:
                marker_first = normalize_match_text(f"{marker} {suffix}")
                if marker_first and marker_first not in variants:
                    variants.append(marker_first)
                if prefix:
                    reordered = normalize_match_text(f"{marker} {suffix} {prefix}")
                    if reordered and reordered not in variants:
                        variants.append(reordered)
                    suffix_then_prefix = normalize_match_text(f"{suffix} {prefix}")
                    if suffix_then_prefix and suffix_then_prefix not in variants:
                        variants.append(suffix_then_prefix)
    core = _question_core(text or "")
    if core and core not in variants:
        variants.append(core)
    # OCR often confuses leading uppercase "I" with lowercase "l" in short prompts.
    for seed in list(variants):
        fixed = re.sub(r"^lle\b", "ile", seed)
        if fixed and fixed not in variants:
            variants.append(fixed)
    return variants


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
    if len(norm) == 1:
        # Keep only plausible single-char options; reject OCR marker artifacts like stray 'o'.
        return bool(norm.isdigit() or norm in {"a", "b", "c", "d"})
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


def _evaluate_registry_quality(payload: Dict[str, Any]) -> Dict[str, Any]:
    registry_items = payload.get("items") if isinstance(payload.get("items"), dict) else {}
    host_scoped_count = 0
    site_key_scoped_count = 0
    canonical_site_id_scoped_count = 0
    for value in registry_items.values():
        if not isinstance(value, dict):
            continue
        if str(value.get("site_host") or "").strip() or (
            isinstance(value.get("site_hosts"), list) and any(str(v or "").strip() for v in value.get("site_hosts") or [])
        ):
            host_scoped_count += 1
        if str(value.get("stable_site_key") or "").strip() or (
            isinstance(value.get("stable_site_keys"), list) and any(str(v or "").strip() for v in value.get("stable_site_keys") or [])
        ):
            site_key_scoped_count += 1
        if str(value.get("canonical_site_id") or "").strip() or (
            isinstance(value.get("canonical_site_ids"), list) and any(str(v or "").strip() for v in value.get("canonical_site_ids") or [])
        ):
            canonical_site_id_scoped_count += 1
    reasons: List[str] = []
    if not registry_items:
        reasons.append("empty_registry")
    if site_key_scoped_count == 0 and canonical_site_id_scoped_count == 0:
        reasons.append("missing_site_family_scope")
    return {
        "is_ready": not reasons,
        "reasons": reasons,
        "signals": {
            "item_count": len(registry_items),
            "host_scoped_count": host_scoped_count,
            "stable_site_key_scoped_count": site_key_scoped_count,
            "canonical_site_id_scoped_count": canonical_site_id_scoped_count,
        },
    }


def _domain_entry_matches_site(entry: Dict[str, Any], host: str, stable_site_key: str, canonical_site_id: str) -> bool:
    normalized_site_key = str(stable_site_key or "").strip().lower()
    normalized_canonical_site_id = str(canonical_site_id or "").strip().lower()
    entry_site_key = str(entry.get("stable_site_key") or "").strip().lower()
    if entry_site_key:
        return entry_site_key == normalized_site_key
    entry_site_keys = entry.get("stable_site_keys") if isinstance(entry.get("stable_site_keys"), list) else []
    allowed_site_keys = [str(value or "").strip().lower() for value in entry_site_keys if str(value or "").strip()]
    if allowed_site_keys:
        return normalized_site_key in allowed_site_keys
    entry_canonical_site_id = str(entry.get("canonical_site_id") or "").strip().lower()
    if entry_canonical_site_id:
        return entry_canonical_site_id == normalized_canonical_site_id
    entry_canonical_site_ids = entry.get("canonical_site_ids") if isinstance(entry.get("canonical_site_ids"), list) else []
    allowed_canonical_site_ids = [str(value or "").strip().lower() for value in entry_canonical_site_ids if str(value or "").strip()]
    if allowed_canonical_site_ids:
        return normalized_canonical_site_id in allowed_canonical_site_ids
    normalized_host = str(host or "").strip().lower()
    site_host = str(entry.get("site_host") or "").strip().lower()
    if site_host:
        return site_host == normalized_host
    site_hosts = entry.get("site_hosts") if isinstance(entry.get("site_hosts"), list) else []
    allowed_hosts = [str(value or "").strip().lower() for value in site_hosts if str(value or "").strip()]
    if allowed_hosts:
        return normalized_host in allowed_hosts
    return True


def _domain_answer_source_path() -> Optional[Path]:
    raw = str(os.environ.get("FULLBOT_DOMAIN_ANSWER_SOURCE_PATH", "") or "").strip()
    return Path(raw) if raw else None


def build_answer_source_policy(*, page_url: str, stable_site_key: str = "", canonical_site_id: str = "") -> Dict[str, Any]:
    host = str(urlparse(page_url or "").hostname or "").strip().lower()
    is_synthetic_host = host in {"127.0.0.1", "localhost"}
    domain_path = _domain_answer_source_path()
    registry_quality = {"is_ready": False, "reasons": ["registry_not_configured"], "signals": {}}
    release_manifest = {
        "release_family": "answer_source_release_manifest",
        "selected_path": "",
        "selected_source": "qa_cache",
        "item_count": 0,
        "content_hash": "",
    }
    if domain_path:
        payload = _load_cache(domain_path)
        registry_quality = _evaluate_registry_quality(payload)
        items = payload.get("items") if isinstance(payload.get("items"), dict) else {}
        try:
            content_hash = hashlib.sha1(domain_path.read_text(encoding="utf-8", errors="ignore").encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            content_hash = ""
        release_manifest = {
            "release_family": "answer_source_release_manifest",
            "selected_path": str(domain_path),
            "selected_source": "domain_registry",
            "item_count": len(items),
            "content_hash": content_hash,
        }
    domain_registry_ready = bool(domain_path) and bool(registry_quality.get("is_ready"))
    preferred_source = "qa_cache" if is_synthetic_host else ("domain_registry" if domain_registry_ready else "qa_cache")
    allowed_sources = ["qa_cache"] if is_synthetic_host else (["domain_registry", "qa_cache"] if domain_registry_ready else ["qa_cache"])
    return {
        "host": host,
        "stable_site_key": str(stable_site_key or "").strip(),
        "canonical_site_id": str(canonical_site_id or "").strip(),
        "is_synthetic_host": is_synthetic_host,
        "preferred_source": preferred_source,
        "allowed_sources": allowed_sources,
        "domain_path": str(domain_path) if domain_path else "",
        "domain_registry_ready": domain_registry_ready,
        "domain_registry_quality_gate": registry_quality,
        "answer_source_release_manifest": release_manifest,
    }


def _load_answer_sources(cache_path: Path, *, page_url: str, stable_site_key: str = "", canonical_site_id: str = "") -> Dict[str, Any]:
    policy = build_answer_source_policy(
        page_url=page_url,
        stable_site_key=stable_site_key,
        canonical_site_id=canonical_site_id,
    )
    qa_payload = _load_cache(cache_path)
    items = qa_payload.get("items") if isinstance(qa_payload.get("items"), dict) else {}
    merged_items: Dict[str, Any] = {}
    if "qa_cache" in policy["allowed_sources"]:
        for key, value in items.items():
            if not isinstance(value, dict):
                continue
            entry = dict(value)
            entry.setdefault("answer_source_kind", "qa_cache")
            merged_items[str(key)] = entry
    domain_path = Path(policy["domain_path"]) if str(policy["domain_path"]).strip() else None
    if domain_path and "domain_registry" in policy["allowed_sources"]:
        domain_payload = _load_cache(domain_path)
        domain_items = domain_payload.get("items") if isinstance(domain_payload.get("items"), dict) else {}
        for key, value in domain_items.items():
            if not isinstance(value, dict):
                continue
            entry = dict(value)
            if not _domain_entry_matches_site(
                entry,
                str(policy.get("host") or ""),
                str(policy.get("stable_site_key") or ""),
                str(policy.get("canonical_site_id") or ""),
            ):
                continue
            entry.setdefault("answer_source_kind", "domain_registry")
            if policy["preferred_source"] == "domain_registry" or str(key) not in merged_items:
                merged_items[str(key)] = entry
    return {"items": merged_items, "source_policy": policy}


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
    page_url: str = "",
    stable_site_key: str = "",
    canonical_site_id: str = "",
) -> ResolvedQuizAnswer:
    cache_payload = _load_answer_sources(
        cache_path,
        page_url=page_url,
        stable_site_key=stable_site_key,
        canonical_site_id=canonical_site_id,
    )
    cache = _build_cache_indexes(cache_payload)
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
        source_prefix = "domain_" if str(best.get("answer_source_kind") or "") == "domain_registry" else ""
        return _resolved_from_entry(best, source=f"{source_prefix}signature", confidence=0.98)

    question_variants = _question_match_variants(active_question)
    norm_question = question_variants[0] if question_variants else ""
    norm_question_core = question_variants[1] if len(question_variants) > 1 else _question_core(active_question)
    best_entry = None
    best_score = 0.0
    best_question_score = 0.0
    for entry in items:
        q_score_full = max(
            (text_similarity(candidate, entry.get("question_text_norm") or "") for candidate in question_variants),
            default=0.0,
        )
        q_score_core = max(
            (text_similarity(candidate, entry.get("question_core_norm") or "") for candidate in question_variants),
            default=0.0,
        )
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
        source_prefix = "domain_" if str(best_entry.get("answer_source_kind") or "") == "domain_registry" else ""
        return _resolved_from_entry(best_entry, source=f"{source_prefix}fuzzy", confidence=min(0.95, best_score))
    if best_entry and not screen_opts:
        entry_qtype = str(best_entry.get("question_type") or "single").strip().lower()
        min_question_only_score = 0.90
        if entry_qtype in {"dropdown", "dropdown_scroll"}:
            min_question_only_score = 0.84
        if extract_arithmetic_answer(active_question) is not None and entry_qtype in _CHOICE_FAMILY:
            min_question_only_score = min(min_question_only_score, 0.80)
        if (entry_qtype in _CHOICE_FAMILY or entry_qtype in {"dropdown", "dropdown_scroll"}) and best_question_score >= min_question_only_score:
            source_prefix = "domain_" if str(best_entry.get("answer_source_kind") or "") == "domain_registry" else ""
            return _resolved_from_entry(
                best_entry,
                source=f"{source_prefix}question_only_cache",
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
        qid_entry = by_qid[str(qid)]
        source_prefix = "domain_" if str(qid_entry.get("answer_source_kind") or "") == "domain_registry" else ""
        return _resolved_from_entry(qid_entry, source=f"{source_prefix}qid_fallback", confidence=0.9)

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
