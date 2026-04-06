from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def _env_flag(name: str, default: str = "1") -> bool:
    raw = str(os.environ.get(name, default) or default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _default_model_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "data" / "models" / "control_kind_classifier_v10.json"


def _model_rollout_contract_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "data" / "models" / "control_kind_rollout.json"


def _resolve_rollout_model_path() -> Path:
    explicit_contract = str(os.environ.get("FULLBOT_CONTROL_KIND_MODEL_ROLLOUT_PATH", "") or "").strip()
    contract_path = Path(explicit_contract) if explicit_contract else _model_rollout_contract_path()
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return _default_model_path()
    if not isinstance(payload, dict):
        return _default_model_path()
    selected_path = str(payload.get("selected_model_path") or "").strip()
    return Path(selected_path) if selected_path else _default_model_path()


def build_explicit_model_selector() -> Dict[str, Any]:
    selector = str(os.environ.get("FULLBOT_CONTROL_KIND_MODEL_SELECTOR", "") or "").strip().lower()
    explicit_path = str(os.environ.get("FULLBOT_CONTROL_KIND_MODEL_PATH", "") or "").strip()
    selected_variant = "default"
    selected_path = _default_model_path()
    source = "default_selector"
    if selector in {"rollout", "latest"}:
        selected_variant = "rollout"
        selected_path = _resolve_rollout_model_path()
        source = "selector_env"
    elif selector in {"path", "custom"}:
        selected_variant = "custom_path"
        selected_path = Path(explicit_path) if explicit_path else _default_model_path()
        source = "selector_env"
    elif explicit_path:
        selected_variant = "legacy_path"
        selected_path = Path(explicit_path)
        source = "legacy_path_env"
    return {
        "selected_variant": selected_variant,
        "selected_path": str(selected_path),
        "source": source,
        "selector": selector or "default",
        "is_explicit": source == "selector_env",
    }


def _load_model() -> Optional[Dict[str, Any]]:
    if not _env_flag("FULLBOT_CONTROL_KIND_MODEL_ENABLED", "1"):
        return None
    selector = build_explicit_model_selector()
    path = Path(str(selector.get("selected_path") or "").strip() or str(_default_model_path()))
    key = str(path).lower()
    if key in _MODEL_CACHE:
        payload = _MODEL_CACHE[key]
        return payload if payload else None
    if not path.exists():
        _MODEL_CACHE[key] = {}
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        _MODEL_CACHE[key] = {}
        return None
    if not isinstance(payload, dict):
        _MODEL_CACHE[key] = {}
        return None
    _MODEL_CACHE[key] = payload
    return payload


def _as_float(v: Any) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _norm_token(v: Any) -> str:
    return str(v or "").strip().lower() or "unknown"


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def _softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    m = max(float(v) for v in logits)
    exps = [math.exp(float(v) - m) for v in logits]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / float(len(exps)) for _ in exps]
    return [float(v) / total for v in exps]


def _extract_features(screen_state: Dict[str, Any], feature_names: Sequence[str]) -> Dict[str, float]:
    questions = screen_state.get("questions") if isinstance(screen_state.get("questions"), list) else []
    blocks = screen_state.get("blocks") if isinstance(screen_state.get("blocks"), list) else []
    active_block = screen_state.get("active_block") if isinstance(screen_state.get("active_block"), dict) else {}
    features = screen_state.get("quiz_type_features") if isinstance(screen_state.get("quiz_type_features"), dict) else {}
    sig = screen_state.get("type_signals") if isinstance(screen_state.get("type_signals"), dict) else {}
    marker = sig.get("marker") if isinstance(sig.get("marker"), dict) else {}
    block_counts = sig.get("block_type_counts") if isinstance(sig.get("block_type_counts"), dict) else {}
    pf = sig.get("artifact_prompt_flags") if isinstance(sig.get("artifact_prompt_flags"), dict) else {}
    qtext = str(screen_state.get("question_text") or "").strip().lower()
    options = screen_state.get("options") if isinstance(screen_state.get("options"), list) else []
    answer_texts: List[str] = []
    for ans in options:
        if not isinstance(ans, dict):
            continue
        txt = str(ans.get("norm_text") or ans.get("text") or "").strip().lower()
        if txt:
            answer_texts.append(txt)
    joined_answers = " | ".join(answer_texts)
    answer_token_counts = [len(t.split()) for t in answer_texts if t]
    answer_numeric_count = sum(1 for t in answer_texts if any(ch.isdigit() for ch in t))
    answer_short_count = sum(1 for c in answer_token_counts if c <= 2)
    answer_long_count = sum(1 for c in answer_token_counts if c >= 5)
    answer_option_label_count = sum(1 for t in answer_texts if "opcja_" in t or "option_" in t or t.startswith("option "))
    question_control_kinds = [str(q.get("control_kind") or "").strip().lower() for q in questions if isinstance(q, dict)]
    question_scroll_count = sum(1 for q in questions if isinstance(q, dict) and q.get("scroll_needed"))
    question_select_count = sum(1 for q in questions if isinstance(q, dict) and q.get("select_bbox"))
    question_input_count = sum(1 for q in questions if isinstance(q, dict) and q.get("input_bbox"))
    denom_answers = float(max(1, len(answer_texts)))
    denom_questions = float(max(1, len(question_control_kinds)))
    prompt_tokens = set(qtext.replace(":", " ").replace("?", " ").split())
    answer_tokens: set[str] = set()
    for txt in answer_texts:
        answer_tokens.update(txt.replace(":", " ").replace("?", " ").split())
    overlap = len(prompt_tokens & answer_tokens) / float(max(1, len(prompt_tokens)))

    dropdown_phrases = ("wybierz z listy", "select from list", "rozwin", "rozwiń", "lista", "dropdown")
    scroll_phrases = ("scroll", "przewin", "expand", "rozszerz", "lista jest dluga", "lista jest długa")
    single_choice_phrases = ("choose one", "one option", "wybierz jedna", "wybierz jedną", "jedna odpowiedz", "jedną odpowiedź")
    multi_choice_phrases = ("choose all", "select all", "zaznacz wszystkie", "co najmniej", "pasujace", "pasujące")
    text_phrases = ("wpisz", "przepisz", "type", "write", "enter", "uzupelnij", "uzupełnij", "podaj", "result", "wynik", "kod", "code")
    slider_phrases = ("suwak", "slider", "przesun", "przesuń", "intensity", "level", "progress", "temperature", "wiek", "age", "rating", "scale")
    answer_dropdown_hints = ("expand", "scroll", "lista jest dluga", "lista jest długa", "rozwin", "rozwiń", "wybierz", "select")
    answer_text_hints = ("pole jest wymagane", "twoja odpowiedz", "twoja odpowiedź", "mozesz wpisac", "enter", "type here", "input")
    answer_slider_hints = ("%", "/10", "/ 10", "lat", "years", "poziom", "level", "intensity", "temperature", "wiek")

    parser_kind = _norm_token(screen_state.get("control_kind"))
    active_block_type = _norm_token(screen_state.get("active_block_type"))
    active_block_family = _norm_token(active_block.get("block_family"))
    active_block_control_kind = _norm_token(active_block.get("control_kind"))
    parser_quiz_type = _norm_token(screen_state.get("detected_quiz_type"))
    has_select = _as_float(screen_state.get("select_bbox") is not None)
    has_input = _as_float(screen_state.get("input_bbox") is not None)
    artifact_scroll = _as_float(pf.get("scroll_hint"))
    artifact_slider = _as_float(pf.get("slider_hint"))
    artifact_multi = _as_float(pf.get("multi_hint"))

    out: Dict[str, float] = {
        "question_count": _as_float(len(questions)),
        "block_count": _as_float(len(blocks)),
        "answer_count": _as_float(len(options)),
        "has_next": _as_float(screen_state.get("next_bbox") is not None),
        "has_select": has_select,
        "has_input": has_input,
        "scroll_needed": _as_float(screen_state.get("scroll_needed")),
        "type_confidence": _as_float(screen_state.get("type_confidence")),
        "decision_margin": _as_float(screen_state.get("decision_margin")),
        "artifact_triple_hint": _as_float(pf.get("triple_hint")),
        "artifact_mix_hint": _as_float(pf.get("mix_hint")),
        "artifact_slider_hint": artifact_slider,
        "artifact_scroll_hint": artifact_scroll,
        "artifact_multi_hint": artifact_multi,
        "qt_has_choose": 1.0 if any(tok in qtext for tok in ("choose", "wybierz")) else 0.0,
        "qt_has_select": 1.0 if any(tok in qtext for tok in ("select", "zaznacz")) else 0.0,
        "qt_has_option": 1.0 if "option" in qtext or "opcja" in qtext else 0.0,
        "qt_has_placeholder_pick": 1.0 if "-- wybierz --" in qtext or "-- select --" in qtext else 0.0,
        "qt_has_type": 1.0 if "type" in qtext else 0.0,
        "qt_has_write": 1.0 if any(tok in qtext for tok in ("write", "wpisz", "przepisz")) else 0.0,
        "qt_has_copy": 1.0 if "copy" in qtext else 0.0,
        "qt_has_code": 1.0 if "code" in qtext or "kod" in qtext else 0.0,
        "qt_has_result": 1.0 if "result" in qtext or "wynik" in qtext else 0.0,
        "qt_has_input_polish": 1.0 if "tutaj" in qtext else 0.0,
        "qt_has_multi_polish": 1.0 if any(tok in qtext for tok in ("wszystkie", "pasujace", "apply")) else 0.0,
        "qt_has_dropdown_phrase": 1.0 if any(tok in qtext for tok in dropdown_phrases) else 0.0,
        "qt_has_scroll_phrase": 1.0 if any(tok in qtext for tok in scroll_phrases) else 0.0,
        "qt_has_single_choice_phrase": 1.0 if any(tok in qtext for tok in single_choice_phrases) else 0.0,
        "qt_has_multi_choice_phrase": 1.0 if any(tok in qtext for tok in multi_choice_phrases) else 0.0,
        "qt_has_text_phrase": 1.0 if any(tok in qtext for tok in text_phrases) else 0.0,
        "qt_has_slider_phrase": 1.0 if any(tok in qtext for tok in slider_phrases) else 0.0,
        "qt_has_percent": 1.0 if "%" in qtext else 0.0,
        "qt_has_scale": 1.0 if any(tok in qtext for tok in ("/10", "1-10", "1/10", "0-10", "0/10")) else 0.0,
        "qt_has_colon": 1.0 if ":" in qtext else 0.0,
        "qt_has_number_token": 1.0 if any(ch.isdigit() for ch in qtext) else 0.0,
        "ans_has_expand": 1.0 if any(tok in joined_answers for tok in answer_dropdown_hints) else 0.0,
        "ans_has_scroll": 1.0 if any(tok in joined_answers for tok in ("scroll", "przewin", "lista jest dluga", "lista jest długa")) else 0.0,
        "ans_has_option_token": 1.0 if answer_option_label_count > 0 else 0.0,
        "ans_has_placeholder_like": 1.0 if any("--" in txt or "select" == txt or "wybierz" == txt for txt in answer_texts) else 0.0,
        "ans_has_text_field_hint": 1.0 if any(tok in joined_answers for tok in answer_text_hints) else 0.0,
        "ans_has_slider_hint": 1.0 if any(tok in joined_answers for tok in answer_slider_hints) else 0.0,
        "ans_count_short_tokens": float(answer_short_count),
        "ans_count_numeric_tokens": float(answer_numeric_count),
        "block_has_select_bbox": 1.0 if active_block.get("select_bbox") else 0.0,
        "block_has_input_bbox": 1.0 if active_block.get("input_bbox") else 0.0,
        "block_has_next_bbox": 1.0 if active_block.get("next_bbox") else 0.0,
        "block_answers_long_ratio": float(answer_long_count) / denom_answers,
        "block_answers_short_ratio": float(answer_short_count) / denom_answers,
        "block_answers_numeric_ratio": float(answer_numeric_count) / denom_answers,
        "block_answers_option_label_ratio": float(answer_option_label_count) / denom_answers,
        "block_prompt_answer_overlap": overlap,
        "questions_any_select": float(question_select_count > 0),
        "questions_any_input": float(question_input_count > 0),
        "questions_any_scroll": float(question_scroll_count > 0),
        "questions_text_ratio": float(sum(1 for k in question_control_kinds if k == "text")) / denom_questions,
        "questions_dropdown_ratio": float(sum(1 for k in question_control_kinds if k == "dropdown")) / denom_questions,
        "questions_slider_ratio": float(sum(1 for k in question_control_kinds if k == "slider")) / denom_questions,
        "sig_question_count": _as_float(sig.get("question_count")),
        "sig_has_next": _as_float(sig.get("has_next")),
        "sig_has_select": _as_float(sig.get("has_select")),
        "sig_has_input": _as_float(sig.get("has_input")),
        "sig_option_count": _as_float(sig.get("option_count")),
        "sig_vertical_regularity": _as_float(sig.get("vertical_regularity")),
        "sig_scroll_needed": _as_float(sig.get("scroll_needed")),
        "sig_marker_total": _as_float(marker.get("marker_total")),
        "sig_marker_circle": _as_float(marker.get("marker_circle")),
        "sig_marker_square": _as_float(marker.get("marker_square")),
        "sig_marker_unknown": _as_float(marker.get("marker_unknown")),
        "sig_marker_mean_conf": _as_float(marker.get("marker_mean_conf")),
        "sig_block_type_count_single": _as_float(block_counts.get("single")),
        "sig_block_type_count_multi": _as_float(block_counts.get("multi")),
        "sig_block_type_count_dropdown": _as_float(block_counts.get("dropdown")),
        "sig_block_type_count_dropdown_scroll": _as_float(block_counts.get("dropdown_scroll")),
        "sig_block_type_count_text": _as_float(block_counts.get("text")),
        "sig_block_type_count_slider": _as_float(block_counts.get("slider")),
        "sig_pf_multi_hint": _as_float(pf.get("multi_hint")),
        "sig_pf_scroll_hint": _as_float(pf.get("scroll_hint")),
        "sig_pf_triple_hint": _as_float(pf.get("triple_hint")),
        "sig_pf_mix_hint": _as_float(pf.get("mix_hint")),
        "sig_pf_text_hint": _as_float(pf.get("text_hint")),
        "sig_pf_slider_hint": _as_float(pf.get("slider_hint")),
    }
    for key, val in features.items():
        out[f"f_{key}"] = _as_float(val)
    out["combo_scroll_choice_like"] = float(
        artifact_scroll > 0.0 and has_select == 0.0 and parser_kind == "choice" and active_block_family == "choice"
    )
    out["combo_scroll_dropdown_like"] = float(
        artifact_scroll > 0.0
        and (
            has_select > 0.0
            or "dropdown" in parser_quiz_type
            or out.get("ans_has_expand", 0.0) > 0.0
            or out.get("ans_has_scroll", 0.0) > 0.0
            or out.get("qt_has_dropdown_phrase", 0.0) > 0.0
        )
    )
    out["combo_text_like"] = float(
        has_input > 0.0 or out.get("qt_has_text_phrase", 0.0) > 0.0 or out.get("ans_has_text_field_hint", 0.0) > 0.0 or active_block_type == "text"
    )
    out["combo_slider_like"] = float(
        artifact_slider > 0.0
        or out.get("qt_has_slider_phrase", 0.0) > 0.0
        or out.get("ans_has_slider_hint", 0.0) > 0.0
        or active_block_type == "slider"
        or active_block_family == "slider"
        or active_block_control_kind == "slider"
        or parser_kind == "slider"
        or parser_quiz_type == "slider"
    )
    out["combo_multi_choice_like"] = float(
        artifact_multi > 0.0 or out.get("qt_has_multi_choice_phrase", 0.0) > 0.0 or active_block_type == "multi"
    )
    out["combo_select_present"] = float(has_select > 0.0 or out.get("block_has_select_bbox", 0.0) > 0.0 or out.get("questions_any_select", 0.0) > 0.0)
    out["combo_input_present"] = float(has_input > 0.0 or out.get("block_has_input_bbox", 0.0) > 0.0 or out.get("questions_any_input", 0.0) > 0.0)
    for name in feature_names:
        out.setdefault(name, 0.0)
    cat_values = {
        "parser_control_kind": parser_kind,
        "active_block_type": active_block_type,
        "active_block_family": active_block_family,
        "active_block_control_kind": active_block_control_kind,
        "parser_detected_quiz_type": parser_quiz_type,
    }
    for name in feature_names:
        if "__" not in name:
            continue
        prefix, cat = name.split("__", 1)
        out[name] = 1.0 if cat_values.get(prefix, "unknown") == cat else 0.0
    return out


def classify_control_kind(screen_state: Dict[str, Any]) -> Dict[str, Any]:
    model = _load_model()
    if not model or not isinstance(screen_state, dict):
        return {"pred": str(screen_state.get("control_kind") or "unknown"), "conf": 0.0, "probs": {}, "source": "disabled"}
    feature_names = model.get("feature_names") if isinstance(model.get("feature_names"), list) else []
    classes = model.get("classes") if isinstance(model.get("classes"), list) else []
    if not feature_names or not classes:
        return {"pred": str(screen_state.get("control_kind") or "unknown"), "conf": 0.0, "probs": {}, "source": "invalid_model"}
    f = _extract_features(screen_state, feature_names)
    x = [float(f.get(name, 0.0)) for name in feature_names]
    mu = model.get("norm_mean") if isinstance(model.get("norm_mean"), list) else [0.0] * len(feature_names)
    sigma = model.get("norm_std") if isinstance(model.get("norm_std"), list) else [1.0] * len(feature_names)
    xn: List[float] = []
    for idx, val in enumerate(x):
        m = float(mu[idx]) if idx < len(mu) else 0.0
        s = float(sigma[idx]) if idx < len(sigma) else 1.0
        if abs(s) < 1e-9:
            s = 1.0
        xn.append((val - m) / s)
    model_type = str(model.get("model_type") or "mlp").strip().lower()
    logits: List[float] = [0.0 for _ in classes]
    if model_type == "mlp":
        W1 = model.get("W1") if isinstance(model.get("W1"), list) else []
        b1 = model.get("b1") if isinstance(model.get("b1"), list) else []
        W2 = model.get("W2") if isinstance(model.get("W2"), list) else []
        b2 = model.get("b2") if isinstance(model.get("b2"), list) else []
        h: List[float] = []
        for j in range(len(b1)):
            acc = float(b1[j] or 0.0)
            for i, xv in enumerate(xn):
                try:
                    acc += xv * float(W1[i][j])
                except Exception:
                    continue
            h.append(_relu(acc))
        for k in range(len(classes)):
            acc = float(b2[k] or 0.0) if k < len(b2) else 0.0
            for j, hv in enumerate(h):
                try:
                    acc += hv * float(W2[j][k])
                except Exception:
                    continue
            logits[k] = acc
    else:
        return {"pred": str(screen_state.get("control_kind") or "unknown"), "conf": 0.0, "probs": {}, "source": "unsupported_model"}
    probs_list = _softmax(logits)
    probs = {str(cls): float(probs_list[idx]) for idx, cls in enumerate(classes)}
    ordered = sorted(probs.items(), key=lambda kv: float(kv[1]), reverse=True)
    pred = ordered[0][0] if ordered else str(screen_state.get("control_kind") or "unknown")
    conf = float(ordered[0][1]) if ordered else 0.0
    return {"pred": pred, "conf": conf, "probs": probs, "source": "mlp_model"}
