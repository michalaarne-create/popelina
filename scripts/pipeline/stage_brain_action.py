from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Optional


def _norm(s: str) -> str:
    return str(s or "").strip().lower()


def _resolve_quiz_cache_path(root: Path) -> Path:
    env_path = str(os.environ.get("FULLBOT_QUIZ_ANSWER_CACHE", "") or "").strip()
    if env_path:
        return Path(env_path)
    preferred = root / "data" / "answers" / "qa_cache.json"
    if preferred.exists():
        return preferred
    return root / "quiz" / "data" / "qa_cache.json"


def _quiz_dom_answer_click(
    *,
    send_click_from_bbox: Callable[..., bool],
    screenshot_path: Path,
    log: Callable[[str], None],
    ensure_dom_fallback: Optional[Callable[[str], bool]] = None,
) -> bool:
    if callable(ensure_dom_fallback):
        if not bool(ensure_dom_fallback("answer")):
            log("[DOM FALLBACK] unavailable: cannot read answer controls.")
            return False
    root = Path(__file__).resolve().parents[2]
    controls_path = root / "scripts" / "dom" / "dom_live" / "current_controls.json"
    qa_cache_path = _resolve_quiz_cache_path(root)
    if not controls_path.exists() or not qa_cache_path.exists():
        return False
    try:
        controls_payload = json.loads(controls_path.read_text(encoding="utf-8", errors="replace"))
        qa_payload = json.loads(qa_cache_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return False
    if not isinstance(controls_payload, dict) or not isinstance(qa_payload, dict):
        return False
    qid = str(((controls_payload.get("meta") or {}).get("qid")) or "")
    if not qid:
        log("[DOM FALLBACK] read controls but qid is empty.")
        return False
    items = qa_payload.get("items") if isinstance(qa_payload.get("items"), dict) else {}
    entry = items.get(qid) if isinstance(items, dict) else None
    if not isinstance(entry, dict):
        return False
    options_text = entry.get("options_text") if isinstance(entry.get("options_text"), dict) else {}
    selected = entry.get("selected_options") if isinstance(entry.get("selected_options"), list) else []
    selected_texts = {_norm(options_text.get(str(k), "")) for k in selected}
    selected_texts.discard("")
    if not selected_texts:
        log(f"[DOM FALLBACK] qid={qid} selected options empty in qa_cache.")
        return False
    log(f"[DOM FALLBACK] read answer: qid={qid} selected={sorted(selected_texts)}")
    controls = controls_payload.get("controls") if isinstance(controls_payload.get("controls"), list) else []
    for ctrl in controls:
        if not isinstance(ctrl, dict):
            continue
        kind = _norm(ctrl.get("kind", ""))
        if kind not in {"radio", "checkbox", "option"}:
            continue
        if bool(ctrl.get("disabled")):
            continue
        bbox = ctrl.get("bbox")
        val = _norm(ctrl.get("value", ""))
        txt = _norm(ctrl.get("text", ""))
        if bbox and (val in selected_texts or txt in selected_texts):
            ok = send_click_from_bbox(bbox, screenshot_path, "[FALLBACK] Quiz DOM fallback answer")
            if ok:
                log(f"[FALLBACK] Quiz DOM fallback clicked answer for {qid}.")
            return bool(ok)
    return False


def _quiz_dom_next_click(
    *,
    send_click_from_bbox: Callable[..., bool],
    screenshot_path: Path,
    log: Callable[[str], None],
    ensure_dom_fallback: Optional[Callable[[str], bool]] = None,
) -> bool:
    if callable(ensure_dom_fallback):
        if not bool(ensure_dom_fallback("next")):
            log("[DOM FALLBACK] unavailable: cannot read next control.")
            return False
    root = Path(__file__).resolve().parents[2]
    controls_path = root / "scripts" / "dom" / "dom_live" / "current_controls.json"
    if not controls_path.exists():
        return False
    try:
        controls_payload = json.loads(controls_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return False
    if not isinstance(controls_payload, dict):
        return False
    qid = str(((controls_payload.get("meta") or {}).get("qid")) or "")
    controls = controls_payload.get("controls") if isinstance(controls_payload.get("controls"), list) else []
    next_labels = []
    for ctrl in controls:
        if not isinstance(ctrl, dict):
            continue
        kind = _norm(ctrl.get("kind", ""))
        if kind != "button":
            continue
        if bool(ctrl.get("disabled")):
            continue
        text = _norm(ctrl.get("text", ""))
        value = _norm(ctrl.get("value", ""))
        if ("next" not in text and "dalej" not in text and "next" not in value and "dalej" not in value):
            continue
        next_labels.append(str(ctrl.get("text") or ctrl.get("value") or "").strip())
        bbox = ctrl.get("bbox")
        if bbox:
            log(f"[DOM FALLBACK] read next: qid={qid or '-'} labels={next_labels or ['<none>']}")
            ok = send_click_from_bbox(bbox, screenshot_path, "[FALLBACK] Quiz DOM fallback next")
            if ok:
                log("[FALLBACK] Quiz DOM fallback clicked next.")
            return bool(ok)
    if next_labels:
        log(f"[DOM FALLBACK] next candidates without bbox: qid={qid or '-'} labels={next_labels}")
    return False


def run_brain_action(
    *,
    decision: Any,
    summary_path: Path,
    screenshot_path: Path,
    send_click_from_bbox: Callable[..., bool],
    scroll_on_box: Callable[..., bool],
    find_screenshot_for_summary: Callable[[Path], Optional[Path]],
    send_best_click: Callable[[Path, Optional[Path]], None],
    send_random_click: Callable[[Path, Path], None],
    send_key: Callable[[str], bool],
    send_key_repeat: Callable[[str, int], bool],
    send_type: Callable[[str], bool],
    send_wait: Callable[[int], bool],
    ensure_dom_fallback: Optional[Callable[[str], bool]],
    log: Callable[[str], None],
    update_overlay_status: Callable[[str], None],
) -> None:
    t0 = time.perf_counter()
    try:
        state = getattr(decision, "screen_state", None) or {}
        trace = getattr(decision, "trace", None) or {}
        resolved = trace.get("resolved") if isinstance(trace, dict) else {}
        control_kind = str(state.get("control_kind") or (resolved or {}).get("question_type") or "unknown")
        detected_type = str(state.get("detected_quiz_type") or control_kind)
        detected_op = str(state.get("detected_operational_type") or "")
        type_conf = float(state.get("type_confidence") or 0.0)
        type_source = str(state.get("type_source") or "screen")
        has_next = int(bool(state.get("next_bbox")))
        options_n = len(state.get("options") or []) if isinstance(state.get("options"), list) else 0
        log(
            "[INFO] Brain recognized quiz: "
            f"type={detected_type} op={detected_op or control_kind} conf={type_conf:.2f} src={type_source} "
            f"has_next={has_next} options={options_n} "
            f"action={getattr(decision, 'recommended_action', 'idle')}"
        )
    except Exception:
        pass
    actions = list(getattr(decision, "actions", None) or [])
    if actions:
        for action in actions:
            if not isinstance(action, dict):
                continue
            kind = str(action.get("kind") or "")
            reason = str(action.get("reason") or kind)
            bbox = action.get("bbox")
            if kind == "screen_click" and bbox:
                send_click_from_bbox(bbox, screenshot_path, f"Brain {reason}")
            elif kind == "screen_scroll":
                scroll_bbox = action.get("scroll_region_bbox") or bbox
                if scroll_bbox:
                    scroll_on_box(
                        scroll_bbox,
                        screenshot_path,
                        f"Brain {reason}",
                        total_notches=int(action.get("amount") or 4),
                        direction=str(action.get("direction") or "down"),
                    )
            elif kind == "key_press":
                send_key(str(action.get("combo") or ""))
            elif kind == "key_repeat":
                send_key_repeat(str(action.get("combo") or ""), int(action.get("repeat") or 1))
            elif kind == "type_text":
                send_type(str(action.get("text") or ""))
            elif kind == "wait":
                send_wait(int(action.get("amount") or action.get("metadata", {}).get("ms") or 0))
            elif kind == "noop":
                log(f"[INFO] Brain noop: {reason}")
        update_overlay_status("quiz actions completed.")
        log(f"[TIMER] stage_brain_action {time.perf_counter() - t0:.3f}s action=batch[{len(actions)}]")
        return

    if decision.recommended_action == "click_cookies_accept" and decision.target_bbox:
        if send_click_from_bbox(decision.target_bbox, screenshot_path, "Brain cookies accept"):
            scroll_on_box(decision.target_bbox, screenshot_path, "Brain cookies scroll", total_notches=10, direction="down")
    elif decision.target_bbox:
        label = "Brain next" if decision.recommended_action == "click_next" else "Brain answer"
        send_click_from_bbox(decision.target_bbox, screenshot_path, label)
    elif decision.recommended_action == "click_next":
        if _quiz_dom_next_click(
            send_click_from_bbox=send_click_from_bbox,
            screenshot_path=screenshot_path,
            log=log,
            ensure_dom_fallback=ensure_dom_fallback,
        ):
            pass
        else:
            screenshot = find_screenshot_for_summary(summary_path) or screenshot_path
            send_best_click(summary_path, screenshot)
    elif decision.recommended_action in ("click_answer", "fallback_random"):
        if not bool(getattr(decision, "brain_state", {}).get("quiz_mode")):
            send_random_click(summary_path, screenshot_path)
        else:
            clicked = _quiz_dom_answer_click(
                send_click_from_bbox=send_click_from_bbox,
                screenshot_path=screenshot_path,
                log=log,
                ensure_dom_fallback=ensure_dom_fallback,
            )
            if clicked:
                _quiz_dom_next_click(
                    send_click_from_bbox=send_click_from_bbox,
                    screenshot_path=screenshot_path,
                    log=log,
                    ensure_dom_fallback=ensure_dom_fallback,
                )
            else:
                # In quiz mode avoid random behavior, but still execute deterministic
                # screen click fallback so the iteration can make progress.
                screenshot = find_screenshot_for_summary(summary_path) or screenshot_path
                send_best_click(summary_path, screenshot)
                log("[FALLBACK] Quiz mode fallback -> deterministic best screen click.")
    else:
        # First fallback for quiz mode: resolve answer from DOM/QA, click by screen bbox.
        if _quiz_dom_answer_click(
            send_click_from_bbox=send_click_from_bbox,
            screenshot_path=screenshot_path,
            log=log,
            ensure_dom_fallback=ensure_dom_fallback,
        ):
            _quiz_dom_next_click(
                send_click_from_bbox=send_click_from_bbox,
                screenshot_path=screenshot_path,
                log=log,
                ensure_dom_fallback=ensure_dom_fallback,
            )
        else:
            screenshot = find_screenshot_for_summary(summary_path) or screenshot_path
            send_best_click(summary_path, screenshot)
            log("[FALLBACK] Brain idle -> deterministic best screen click.")
    update_overlay_status("rating completed.")
    log(f"[TIMER] stage_brain_action {time.perf_counter() - t0:.3f}s action={decision.recommended_action}")
