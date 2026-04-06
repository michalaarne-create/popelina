from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


BASE_CLASSES: List[str] = ["choice", "dropdown", "text", "slider"]
FEATURE_NAMES: List[str] = [
    "question_count",
    "block_count",
    "answer_count",
    "has_next",
    "has_select",
    "has_input",
    "scroll_needed",
    "type_confidence",
    "decision_margin",
    "artifact_triple_hint",
    "artifact_mix_hint",
    "artifact_slider_hint",
    "artifact_scroll_hint",
    "artifact_multi_hint",
    "f_question_count",
    "f_has_next",
    "f_has_select",
    "f_has_input",
    "f_option_count",
    "f_vertical_regularity",
    "f_scroll_needed",
    "f_multi_hint",
    "f_scroll_hint",
    "f_triple_hint",
    "f_mix_hint",
    "f_text_hint",
    "f_slider_hint",
    "f_marker_total",
    "f_marker_circle",
    "f_marker_square",
    "f_marker_unknown",
    "f_marker_mean_conf",
    "f_is_multi_block_layout",
    "f_is_exact_triple_count",
    "f_distinct_block_type_count",
    "f_all_blocks_same_type",
    "f_has_mixed_controls",
    "f_avg_options_per_block",
    "f_min_options_per_block",
    "f_max_options_per_block",
    "f_single_block_ratio",
    "f_multi_block_ratio",
    "f_dropdown_block_ratio",
    "f_text_block_ratio",
    "f_slider_block_ratio",
    "f_has_slider",
    "f_all_blocks_have_triple_marker",
    "f_all_blocks_have_mix_marker",
    "sig_question_count",
    "sig_has_next",
    "sig_has_select",
    "sig_has_input",
    "sig_option_count",
    "sig_vertical_regularity",
    "sig_scroll_needed",
    "sig_marker_total",
    "sig_marker_circle",
    "sig_marker_square",
    "sig_marker_unknown",
    "sig_marker_mean_conf",
    "sig_block_type_count_single",
    "sig_block_type_count_multi",
    "sig_block_type_count_dropdown",
    "sig_block_type_count_dropdown_scroll",
    "sig_block_type_count_text",
    "sig_block_type_count_slider",
    "sig_pf_multi_hint",
    "sig_pf_scroll_hint",
    "sig_pf_triple_hint",
    "sig_pf_mix_hint",
    "sig_pf_text_hint",
    "sig_pf_slider_hint",
]
CATEGORICAL_FEATURES: Dict[str, List[str]] = {
    "parser_control_kind": ["choice", "dropdown", "text", "slider", "unknown"],
    "active_block_type": ["single", "multi", "dropdown", "dropdown_scroll", "text", "slider", "unknown"],
    "active_block_family": ["choice", "dropdown", "text", "slider", "dropdown_scroll", "unknown"],
    "active_block_control_kind": ["choice", "dropdown", "text", "slider", "unknown"],
    "parser_detected_quiz_type": ["single", "multi", "dropdown", "dropdown_scroll", "text", "slider", "triple", "mixed", "unknown"],
}
TEXT_SIGNAL_FEATURES: List[str] = [
    "qt_has_choose",
    "qt_has_select",
    "qt_has_option",
    "qt_has_placeholder_pick",
    "qt_has_type",
    "qt_has_write",
    "qt_has_copy",
    "qt_has_code",
    "qt_has_result",
    "qt_has_input_polish",
    "qt_has_multi_polish",
    "qt_has_dropdown_phrase",
    "qt_has_scroll_phrase",
    "qt_has_single_choice_phrase",
    "qt_has_multi_choice_phrase",
    "qt_has_text_phrase",
    "qt_has_slider_phrase",
    "qt_has_percent",
    "qt_has_scale",
    "qt_has_colon",
    "qt_has_number_token",
    "ans_has_expand",
    "ans_has_scroll",
    "ans_has_option_token",
    "ans_has_placeholder_like",
    "ans_has_text_field_hint",
    "ans_has_slider_hint",
    "ans_count_short_tokens",
    "ans_count_numeric_tokens",
    "block_has_select_bbox",
    "block_has_input_bbox",
    "block_has_next_bbox",
    "block_answers_long_ratio",
    "block_answers_short_ratio",
    "block_answers_numeric_ratio",
    "block_answers_option_label_ratio",
    "block_prompt_answer_overlap",
    "questions_any_select",
    "questions_any_input",
    "questions_any_scroll",
    "questions_text_ratio",
    "questions_dropdown_ratio",
    "questions_slider_ratio",
    "combo_scroll_choice_like",
    "combo_scroll_dropdown_like",
    "combo_text_like",
    "combo_slider_like",
    "combo_multi_choice_like",
    "combo_select_present",
    "combo_input_present",
]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = line.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _as_float(v: Any) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _norm_token(v: Any) -> str:
    return str(v or "").strip().lower() or "unknown"


def _build_feature_names() -> List[str]:
    names = list(FEATURE_NAMES)
    names.extend(TEXT_SIGNAL_FEATURES)
    for field, cats in CATEGORICAL_FEATURES.items():
        for cat in cats:
            names.append(f"{field}__{cat}")
    return names


def _extract_features(row: Dict[str, Any], feature_names: Sequence[str]) -> Dict[str, float]:
    f = row.get("quiz_type_features") if isinstance(row.get("quiz_type_features"), dict) else {}
    sig = row.get("type_signals") if isinstance(row.get("type_signals"), dict) else {}
    marker = sig.get("marker") if isinstance(sig.get("marker"), dict) else {}
    block_counts = sig.get("block_type_counts") if isinstance(sig.get("block_type_counts"), dict) else {}
    pf = sig.get("artifact_prompt_flags") if isinstance(sig.get("artifact_prompt_flags"), dict) else {}
    qtext = str(row.get("question_text") or "").strip().lower()
    questions = row.get("questions") if isinstance(row.get("questions"), list) else []
    blocks = row.get("blocks") if isinstance(row.get("blocks"), list) else []
    active_block = None
    for block in blocks:
        if isinstance(block, dict) and block.get("is_active"):
            active_block = block
            break
    if active_block is None and blocks and isinstance(blocks[0], dict):
        active_block = blocks[0]
    active_answers = active_block.get("answers") if isinstance(active_block, dict) and isinstance(active_block.get("answers"), list) else []
    answer_texts: List[str] = []
    for ans in active_answers:
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
    qset = set(question_control_kinds)
    denom_answers = float(max(1, len(answer_texts)))
    denom_questions = float(max(1, len(question_control_kinds)))

    dropdown_phrases = (
        "wybierz z listy",
        "select from list",
        "rozwin",
        "rozwiń",
        "lista",
        "dropdown",
    )
    scroll_phrases = ("scroll", "przewin", "expand", "rozszerz", "lista jest dluga", "lista jest długa")
    single_choice_phrases = ("choose one", "one option", "wybierz jedna", "wybierz jedną", "jedna odpowiedz", "jedną odpowiedź")
    multi_choice_phrases = ("choose all", "select all", "zaznacz wszystkie", "co najmniej", "pasujace", "pasujące")
    text_phrases = ("wpisz", "przepisz", "type", "write", "enter", "uzupelnij", "uzupełnij", "podaj", "result", "wynik", "kod", "code")
    slider_phrases = ("suwak", "slider", "przesun", "przesuń", "intensity", "level", "progress", "temperature", "wiek", "age", "rating", "scale")
    answer_dropdown_hints = ("expand", "scroll", "lista jest dluga", "lista jest długa", "rozwin", "rozwiń", "wybierz", "select")
    answer_text_hints = ("pole jest wymagane", "twoja odpowiedz", "twoja odpowiedź", "moz esz wpisac", "mozesz wpisac", "enter", "type here", "input")
    answer_slider_hints = ("%", "/10", "/ 10", "lat", "years", "poziom", "level", "intensity", "temperature", "wiek")
    prompt_tokens = set(qtext.replace(":", " ").replace("?", " ").split())
    answer_tokens: set[str] = set()
    for txt in answer_texts:
        answer_tokens.update(txt.replace(":", " ").replace("?", " ").split())
    overlap = len(prompt_tokens & answer_tokens) / float(max(1, len(prompt_tokens)))
    parser_kind = _norm_token(row.get("parser_control_kind"))
    active_block_type = _norm_token(row.get("active_block_type"))
    active_block_family = _norm_token(row.get("active_block_family"))
    parser_quiz_type = _norm_token(row.get("parser_detected_quiz_type"))
    has_select = _as_float(row.get("has_select"))
    has_input = _as_float(row.get("has_input"))
    artifact_scroll = _as_float(row.get("artifact_scroll_hint"))
    artifact_slider = _as_float(row.get("artifact_slider_hint"))
    artifact_multi = _as_float(row.get("artifact_multi_hint"))
    out: Dict[str, float] = {
        "question_count": _as_float(row.get("question_count")),
        "block_count": _as_float(row.get("block_count")),
        "answer_count": _as_float(row.get("answer_count")),
        "has_next": _as_float(row.get("has_next")),
        "has_select": has_select,
        "has_input": has_input,
        "scroll_needed": _as_float(row.get("scroll_needed")),
        "type_confidence": _as_float(row.get("type_confidence")),
        "decision_margin": _as_float(row.get("decision_margin")),
        "artifact_triple_hint": _as_float(row.get("artifact_triple_hint")),
        "artifact_mix_hint": _as_float(row.get("artifact_mix_hint")),
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
        "block_has_select_bbox": 1.0 if isinstance(active_block, dict) and active_block.get("select_bbox") else 0.0,
        "block_has_input_bbox": 1.0 if isinstance(active_block, dict) and active_block.get("input_bbox") else 0.0,
        "block_has_next_bbox": 1.0 if isinstance(active_block, dict) and active_block.get("next_bbox") else 0.0,
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
    out["combo_scroll_choice_like"] = float(
        artifact_scroll > 0.0
        and has_select == 0.0
        and parser_kind == "choice"
        and active_block_family == "choice"
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
        has_input > 0.0
        or out.get("qt_has_text_phrase", 0.0) > 0.0
        or out.get("ans_has_text_field_hint", 0.0) > 0.0
        or active_block_type == "text"
    )
    out["combo_slider_like"] = float(
        artifact_slider > 0.0
        or out.get("qt_has_slider_phrase", 0.0) > 0.0
        or out.get("ans_has_slider_hint", 0.0) > 0.0
        or active_block_type == "slider"
        or active_block_family == "slider"
        or parser_kind == "slider"
        or parser_quiz_type == "slider"
    )
    out["combo_multi_choice_like"] = float(
        artifact_multi > 0.0
        or out.get("qt_has_multi_choice_phrase", 0.0) > 0.0
        or active_block_type == "multi"
    )
    out["combo_select_present"] = float(
        has_select > 0.0
        or out.get("block_has_select_bbox", 0.0) > 0.0
        or out.get("questions_any_select", 0.0) > 0.0
    )
    out["combo_input_present"] = float(
        has_input > 0.0
        or out.get("block_has_input_bbox", 0.0) > 0.0
        or out.get("questions_any_input", 0.0) > 0.0
    )
    for name in FEATURE_NAMES:
        if name.startswith("f_"):
            out[name] = _as_float(f.get(name[2:]))
    for field, cats in CATEGORICAL_FEATURES.items():
        token = _norm_token(row.get(field))
        for cat in cats:
            out[f"{field}__{cat}"] = 1.0 if token == cat else 0.0
    for name in feature_names:
        out.setdefault(name, 0.0)
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(np.sum(e, axis=1, keepdims=True), 1e-12, None)


def _one_hot(y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((y.shape[0], k), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def build_rollout_payload(*, out_model: Path, out_payload: Dict[str, Any]) -> Dict[str, Any]:
    training_meta = out_payload.get("training_meta") if isinstance(out_payload.get("training_meta"), dict) else {}
    return {
        "release_family": "control_kind_rollout",
        "selected_model_path": str(out_model),
        "selected_variant": "rollout",
        "model_type": str(out_payload.get("model_type") or ""),
        "classes": list(out_payload.get("classes") or []),
        "samples": int(training_meta.get("samples") or 0),
        "val_accuracy": float(training_meta.get("val_accuracy") or 0.0),
        "val_macro_f1": float(training_meta.get("val_macro_f1") or 0.0),
    }


def _forward_mlp(X: np.ndarray, params: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z1 = X @ params["W1"] + params["b1"]
    h1 = _relu(z1)
    logits = h1 @ params["W2"] + params["b2"]
    return z1, h1, logits


def _acc(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y)) if y.size else 0.0


def _macro_f1(pred: np.ndarray, y: np.ndarray, classes: int) -> float:
    vals: List[float] = []
    for idx in range(classes):
        tp = int(np.sum((pred == idx) & (y == idx)))
        fp = int(np.sum((pred == idx) & (y != idx)))
        fn = int(np.sum((pred != idx) & (y == idx)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        vals.append((2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0)
    return float(sum(vals) / max(1, len(vals)))


def _split_indices(rows: Sequence[Dict[str, Any]], classes: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    class_set = set(classes)
    by_class: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        label = _norm_token(row.get("expected_control_kind"))
        if label not in class_set:
            continue
        by_class.setdefault(label, []).append(i)
    rng = random.Random(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for label, idxs in by_class.items():
        rng.shuffle(idxs)
        if len(idxs) <= 1:
            train_idx.extend(idxs)
            continue
        n_val = max(1, int(round(len(idxs) * val_ratio)))
        if n_val >= len(idxs):
            n_val = len(idxs) - 1
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
    out_dim: int,
    class_weights: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    rng = np.random.default_rng(seed)
    in_dim = X_train.shape[1]
    params = {
        "W1": rng.normal(0.0, 0.15, size=(in_dim, hidden)),
        "b1": np.zeros((1, hidden), dtype=np.float64),
        "W2": rng.normal(0.0, 0.15, size=(hidden, out_dim)),
        "b2": np.zeros((1, out_dim), dtype=np.float64),
    }
    y_one = _one_hot(y_train, out_dim)
    best = {k: v.copy() for k, v in params.items()}
    best_acc = -1.0
    best_macro = -1.0
    wait = 0
    patience = 80
    for _ in range(max(1, epochs)):
        z1, h1, logits = _forward_mlp(X_train, params)
        probs = _softmax(logits)
        dlogits = (probs - y_one) * class_weights[y_train][:, None] / max(1, X_train.shape[0])
        dW2 = h1.T @ dlogits
        db2 = np.sum(dlogits, axis=0, keepdims=True)
        dh1 = dlogits @ params["W2"].T
        dz1 = dh1 * (z1 > 0.0)
        dW1 = X_train.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        params["W1"] -= lr * dW1
        params["b1"] -= lr * db1
        params["W2"] -= lr * dW2
        params["b2"] -= lr * db2
        _, _, val_logits = _forward_mlp(X_val, params)
        val_probs = _softmax(val_logits)
        val_pred = np.argmax(val_probs, axis=1)
        acc = _acc(val_pred, y_val)
        macro = _macro_f1(val_pred, y_val, out_dim)
        if (acc > best_acc + 1e-9) or (abs(acc - best_acc) <= 1e-9 and macro > best_macro):
            best = {k: v.copy() for k, v in params.items()}
            best_acc = acc
            best_macro = macro
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best, {"val_acc": float(best_acc), "val_macro_f1": float(best_macro)}


def _classification_report(pred: np.ndarray, y: np.ndarray, classes: Sequence[str]) -> Dict[str, Any]:
    per_class: Dict[str, Any] = {}
    confusion: Dict[str, Dict[str, int]] = {c: {cc: 0 for cc in classes} for c in classes}
    for truth, guess in zip(y.tolist(), pred.tolist()):
        confusion[classes[int(truth)]][classes[int(guess)]] += 1
    for idx, cls in enumerate(classes):
        tp = int(np.sum((pred == idx) & (y == idx)))
        fp = int(np.sum((pred == idx) & (y != idx)))
        fn = int(np.sum((pred != idx) & (y == idx)))
        support = int(np.sum(y == idx))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
    return {"per_class": per_class, "confusion": confusion}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train control_kind classifier from parsed screen benchmark JSONL.")
    ap.add_argument("--data-jsonl", required=True)
    ap.add_argument("--out-model", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--out-rollout", default="")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--val-ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    data_path = Path(args.data_jsonl).resolve()
    rows = _read_jsonl(data_path)
    classes = [c for c in BASE_CLASSES if any(_norm_token(r.get("expected_control_kind")) == c for r in rows)]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    feature_names = _build_feature_names()
    rows = [r for r in rows if _norm_token(r.get("expected_control_kind")) in class_to_idx]
    if len(rows) < 4:
        print("[ERROR] not enough labeled rows")
        return 2

    feats = [_extract_features(r, feature_names) for r in rows]
    X = np.asarray([[float(f.get(name, 0.0)) for name in feature_names] for f in feats], dtype=np.float64)
    y = np.asarray([class_to_idx[_norm_token(r.get("expected_control_kind"))] for r in rows], dtype=np.int64)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    Xn = (X - mu) / sigma

    train_idx, val_idx = _split_indices(rows, classes, args.val_ratio, args.seed)
    X_train = Xn[train_idx]
    y_train = y[train_idx]
    X_val = Xn[val_idx]
    y_val = y[val_idx]
    if X_val.shape[0] == 0:
        print("[ERROR] empty validation split")
        return 2

    counts = np.bincount(y_train, minlength=len(classes)).astype(np.float64)
    class_weights = np.ones(len(classes), dtype=np.float64)
    nonzero = counts > 0
    class_weights[nonzero] = counts.sum() / (len(classes) * counts[nonzero])
    candidates = [
        (int(args.hidden), float(args.lr)),
        (max(16, int(args.hidden // 2)), float(args.lr)),
        (max(16, int(args.hidden)), float(args.lr * 0.5)),
        (max(24, int(args.hidden * 2)), float(args.lr * 0.5)),
        (max(32, int(args.hidden * 3)), float(args.lr * 0.25)),
        (max(64, int(args.hidden * 4)), float(args.lr * 0.2)),
    ]
    tried = set()
    best_params: Dict[str, np.ndarray] | None = None
    best_metrics: Dict[str, float] | None = None
    best_pred: np.ndarray | None = None
    sweep: List[Dict[str, float]] = []
    for hidden, lr in candidates:
        key = (hidden, round(lr, 8))
        if key in tried:
            continue
        tried.add(key)
        params, metrics = train_mlp(
            X_train,
            y_train,
            X_val,
            y_val,
            hidden=hidden,
            epochs=args.epochs,
            lr=lr,
            seed=args.seed,
            out_dim=len(classes),
            class_weights=class_weights,
        )
        _, _, logits_val = _forward_mlp(X_val, params)
        probs_val = _softmax(logits_val)
        pred_val = np.argmax(probs_val, axis=1)
        acc = _acc(pred_val, y_val)
        macro = _macro_f1(pred_val, y_val, len(classes))
        sweep.append({"hidden": hidden, "lr": lr, "val_accuracy": acc, "val_macro_f1": macro})
        if best_metrics is None or acc > best_metrics["val_acc"] + 1e-9 or (abs(acc - best_metrics["val_acc"]) <= 1e-9 and macro > best_metrics["val_macro_f1"]):
            best_params = params
            best_metrics = {"val_acc": acc, "val_macro_f1": macro, "hidden": float(hidden), "lr": lr}
            best_pred = pred_val
    assert best_params is not None and best_metrics is not None and best_pred is not None
    report = {
        "data_jsonl": str(data_path),
        "samples": len(rows),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "classes": classes,
        "feature_names": feature_names,
        "metrics": {
            "val_accuracy": _acc(best_pred, y_val),
            "val_macro_f1": _macro_f1(best_pred, y_val, len(classes)),
        },
        "split": {
            "train_indices": train_idx,
            "val_indices": val_idx,
        },
        "sweep": sweep,
    }
    report.update(_classification_report(best_pred, y_val, classes))

    model = {
        "model_type": "mlp",
        "description": "MLP control_kind classifier trained from parsed screen benchmark features.",
        "classes": classes,
        "feature_names": feature_names,
        "hidden": int(best_metrics["hidden"]),
        "norm_mean": mu.tolist(),
        "norm_std": sigma.tolist(),
        "W1": best_params["W1"].tolist(),
        "b1": best_params["b1"].reshape(-1).tolist(),
        "W2": best_params["W2"].tolist(),
        "b2": best_params["b2"].reshape(-1).tolist(),
        "training_meta": {
            "samples": len(rows),
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "val_accuracy": report["metrics"]["val_accuracy"],
            "val_macro_f1": report["metrics"]["val_macro_f1"],
            "seed": int(args.seed),
            "hidden": int(best_metrics["hidden"]),
            "epochs": int(args.epochs),
            "lr": float(best_metrics["lr"]),
        },
    }

    out_model = Path(args.out_model).resolve()
    out_report = Path(args.out_report).resolve()
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if str(args.out_rollout or "").strip():
        out_rollout = Path(str(args.out_rollout)).resolve()
        out_rollout.parent.mkdir(parents=True, exist_ok=True)
        out_rollout.write_text(
            json.dumps(build_rollout_payload(out_model=out_model, out_payload=model), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps({"out_model": str(out_model), "out_report": str(out_report), "val_accuracy": report["metrics"]["val_accuracy"], "val_macro_f1": report["metrics"]["val_macro_f1"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
