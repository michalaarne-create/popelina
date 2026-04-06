from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from scripts.pipeline.contracts import SCHEMA_VERSION, build_action_id, build_target_instance_id

from .canonical_type_contract import build_canonical_type_contract, canonical_operational_type
from .dom_assist_policy import infer_quiz_type_from_controls as _infer_quiz_type_from_controls_impl
from .agent_state_io import load_json, load_state, save_state
from .agent_targets import select_target
from .decision_core import build_decision_core
from .pipeline_state_builder import build_brain_state
from .quiz_state_builder import build_quiz_state


def _default_logger(message: str) -> None:
    print(message)


def _norm_host(url: str) -> str:
    try:
        return str(urlparse(str(url or "")).netloc or "").strip().lower()
    except Exception:
        return ""


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _update_answered_signature_memory(
    *,
    prev_state: Dict[str, Any],
    transition: Dict[str, Any],
    screen_state: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    memory = dict((prev_state or {}).get("answered_signature_memory") or {})
    prev_action = (prev_state or {}).get("last_action") if isinstance((prev_state or {}).get("last_action"), dict) else {}
    prev_question_signature = str(prev_action.get("question_signature") or "")
    transition_kind = str((transition or {}).get("transition_kind") or "")
    previous_action_kind = str((transition or {}).get("previous_action_kind") or "")
    success = bool((transition or {}).get("success"))
    if prev_question_signature and previous_action_kind in {"answer", "dropdown", "type", "next"} and (
        success or transition_kind in {"new_question", "same_question_answered", "screen_changed", "state_changed"}
    ):
        existing = memory.get(prev_question_signature) if isinstance(memory.get(prev_question_signature), dict) else {}
        memory[prev_question_signature] = {
            "count": int(existing.get("count") or 0) + 1,
            "last_transition_kind": transition_kind,
            "last_seen_ts": time.time(),
            "page_signature": str(prev_action.get("page_signature") or ""),
            "screen_signature": str((prev_state or {}).get("last_screen_signature") or ""),
        }
    current_signature = str(screen_state.get("active_question_signature") or "")
    revisit = bool(current_signature and current_signature in memory)
    revisit_entry = memory.get(current_signature) if revisit and isinstance(memory.get(current_signature), dict) else {}
    flags = {
        "revisit_detected": revisit,
        "revisit_count": int(revisit_entry.get("count") or 0),
        "revisit_signature": current_signature if revisit else "",
    }
    return memory, flags


def _infer_quiz_type_from_controls(
    controls_data: Optional[Dict[str, Any]],
    *,
    screen_state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    return _infer_quiz_type_from_controls_impl(
        controls_data,
        screen_state=screen_state,
    )


def _apply_quiz_type_policy(
    *,
    screen_state: Dict[str, Any],
    controls_data: Optional[Dict[str, Any]],
    prev_state: Dict[str, Any],
) -> Dict[str, Any]:
    out = dict(screen_state or {})
    detected = str(out.get("detected_quiz_type") or out.get("control_kind") or "single")
    op = str(out.get("detected_operational_type") or "")
    conf = float(out.get("type_confidence") or 0.0)
    source = str(out.get("type_source") or "screen")
    dom_type_assist = str(os.environ.get("FULLBOT_QUIZ_TYPE_DOM_ASSIST", "0") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    try:
        conf_min = float(
            os.environ.get(
                "FULLBOT_QUIZ_TYPE_CONF_MIN",
                os.environ.get("FULLBOT_QUIZ_TYPE_MIN_CONF", "0.45"),
            )
            or 0.45
        )
    except Exception:
        conf_min = 0.45

    if conf < conf_min and dom_type_assist:
        dom = _infer_quiz_type_from_controls(controls_data, screen_state=out)
        if isinstance(dom, dict):
            detected = str(dom.get("detected_quiz_type") or detected)
            op = str(dom.get("detected_operational_type") or op)
            conf = float(dom.get("type_confidence") or conf)
            source = str(dom.get("type_source") or "dom_fallback")
            out["type_signals"] = dom.get("type_signals") or out.get("type_signals") or {}

    prev_sig = str((prev_state or {}).get("question_hash") or "")
    cur_sig = str(out.get("active_question_signature") or "")
    prev_type = str((prev_state or {}).get("detected_quiz_type") or "").strip()
    if conf < conf_min and prev_type and prev_sig and cur_sig and prev_sig == cur_sig:
        detected = prev_type
        if not op:
            op = canonical_operational_type(detected)
        conf = max(conf, 0.8)
        source = "sticky_prev"

    if not op:
        op = canonical_operational_type(detected)

    out["detected_quiz_type"] = detected
    out["detected_operational_type"] = op
    out["canonical_type_contract"] = build_canonical_type_contract(
        detected_quiz_type=detected,
        control_kind=str(out.get("control_kind") or ""),
        active_block_type=str(out.get("active_block_type") or ""),
        source=source,
    )
    out["type_confidence"] = float(round(conf, 4))
    out["type_source"] = source
    if "decision_margin" not in out:
        out["decision_margin"] = 0.0
    return out


@dataclass
class BrainDecision:
    recommended_action: str
    target_bbox: Optional[List[float]]
    target_element: Optional[Dict[str, Any]]
    requires_action: bool
    brain_state: dict
    question_data: Optional[dict]
    summary_data: Optional[dict]
    actions: List[Dict[str, Any]] = field(default_factory=list)
    screen_state: Optional[Dict[str, Any]] = None
    answer_source: Optional[str] = None
    fallback_used: bool = False
    page_signature: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None


class PipelineBrainAgent:
    def __init__(
        self,
        question_path: Path,
        state_path: Path,
        logger: Optional[Callable[[str], None]] = None,
        *,
        controls_path: Optional[Path] = None,
        page_path: Optional[Path] = None,
        quiz_answer_cache: Optional[Path] = None,
        current_run_dir: Optional[Path] = None,
    ):
        self.question_path = question_path
        self.state_path = state_path
        self.controls_path = controls_path
        self.page_path = page_path
        self.quiz_answer_cache = quiz_answer_cache or (Path(__file__).resolve().parents[3] / "quiz" / "data" / "qa_cache.json")
        self.current_run_dir = current_run_dir
        self.logger = logger or _default_logger

    def decide(
        self,
        summary_path: Path,
        *,
        region_json_path: Optional[Path] = None,
        screenshot_path: Optional[Path] = None,
    ) -> BrainDecision:
        if self._quiz_mode_enabled():
            return self._decide_quiz(
                summary_path,
                region_json_path=region_json_path,
                screenshot_path=screenshot_path,
            )
        return self._decide_legacy(summary_path)

    def load_state(self) -> dict:
        return load_state(self.state_path, self._log)

    def _decide_legacy(self, summary_path: Path) -> BrainDecision:
        summary_data = load_json(summary_path, self._log)
        question_data = load_json(self.question_path, self._log)
        prev_state = load_state(self.state_path, self._log)
        if summary_data is None:
            self._error(f"Summary missing at {summary_path}; switching to idle.")
            return BrainDecision("idle", None, None, False, prev_state, question_data, None)

        consistency = self._screen_site_consistency()
        if consistency is not None and consistency < 0.5:
            state_dump = dict(prev_state or {})
            state_dump.update({"screen_site_consistency": float(consistency), "screen_site_matched": False})
            save_state(self.state_path, state_dump, self._log)
            return BrainDecision("idle", None, None, False, state_dump, question_data, summary_data)

        try:
            brain = build_brain_state(question_data, summary_data, prev_state, False, False)
        except Exception as exc:
            self._error(f"build_brain_state failed: {exc}; switching to idle.")
            return BrainDecision("idle", None, None, False, prev_state, question_data, summary_data)
        if not isinstance(brain, dict):
            self._error("build_brain_state returned non-dict; switching to idle.")
            return BrainDecision("idle", None, None, False, prev_state, question_data, summary_data)

        action = brain.get("recommended_action", "idle")
        target_element, target_bbox = select_target(brain, action)
        mark_answer = action == "click_answer" and target_bbox is not None
        mark_next = action == "click_next" and target_bbox is not None
        updated = build_brain_state(question_data, summary_data, prev_state, mark_answer, mark_next)
        save_state(self.state_path, updated, self._log)
        requires_action = action in {"click_answer", "click_next", "click_cookies_accept", "scroll_page_down"}
        return BrainDecision(action, target_bbox, target_element, requires_action, updated, question_data, summary_data)

    def _decide_quiz(
        self,
        summary_path: Path,
        *,
        region_json_path: Optional[Path],
        screenshot_path: Optional[Path],
    ) -> BrainDecision:
        prev_state = load_state(self.state_path, self._log)
        bundle = build_quiz_state(
            summary_path=summary_path,
            region_json_path=region_json_path,
            screenshot_path=screenshot_path,
            question_path=self.question_path,
            controls_path=self.controls_path,
            page_path=self.page_path,
        )
        screen_state = bundle.get("screen_state") or {}
        controls_data = bundle.get("controls_data")
        page_data = bundle.get("page_data")
        summary_data = bundle.get("summary_data")
        question_data = bundle.get("question_data")
        screen_state = _apply_quiz_type_policy(
            screen_state=screen_state,
            controls_data=controls_data,
            prev_state=prev_state,
        )
        self._log_quiz_type_decision(screen_state=screen_state)

        core = build_decision_core(
            cache_path=self._quiz_answer_cache(),
            screen_state=screen_state,
            prev_state=prev_state,
            controls_data=controls_data,
            page_data=page_data,
        )
        transition = core.transition
        resolved = core.resolved
        actions = core.actions
        trace = core.trace
        fallback_used = core.fallback_used
        legacy_action = core.legacy_action
        self._log_action_chain(screen_state=screen_state, trace=trace, actions=actions)
        self._log_verify(transition=transition, trace=trace)
        target_element = None
        target_bbox = None
        if actions:
            first = actions[0]
            bbox = first.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                target_bbox = [float(v) for v in bbox]
                target_element = {
                    "kind": first.get("kind"),
                    "reason": first.get("reason"),
                    "bbox": target_bbox,
                }
        requires_action = bool(actions) and legacy_action != "idle"

        updated = self._build_quiz_brain_state(
            prev_state=prev_state,
            screen_state=screen_state,
            question_data=question_data,
            summary_data=summary_data,
            controls_data=controls_data,
            resolved_answer=resolved.to_dict(),
            actions=actions,
            legacy_action=legacy_action,
            trace=trace,
            transition=transition,
            fallback_used=bool(fallback_used),
            page_data=page_data,
            target_bbox=target_bbox,
        )
        self._write_quiz_debug(screen_state=screen_state, trace=trace, resolved=resolved.to_dict())
        save_state(self.state_path, updated, self._log)

        return BrainDecision(
            legacy_action,
            target_bbox,
            target_element,
            requires_action,
            updated,
            question_data,
            summary_data,
            actions=actions,
            screen_state=screen_state,
            answer_source=resolved.source,
            fallback_used=bool(fallback_used),
            page_signature=str(screen_state.get("page_signature") or ""),
            trace={
                **trace,
                "transition": transition,
                "resolved": resolved.to_dict(),
            },
        )

    def _log_quiz_type_decision(self, *, screen_state: Dict[str, Any]) -> None:
        qtype = str(screen_state.get("detected_quiz_type") or "unknown")
        op = str(screen_state.get("detected_operational_type") or "unknown")
        source = str(screen_state.get("type_source") or "unknown")
        layout_type = str(screen_state.get("layout_type") or "single_block")
        active_block_type = str(screen_state.get("active_block_type") or qtype)
        try:
            conf = float(screen_state.get("type_confidence") or 0.0)
        except Exception:
            conf = 0.0
        try:
            margin = float(screen_state.get("decision_margin") or 0.0)
        except Exception:
            margin = 0.0
        probs = screen_state.get("type_probs") if isinstance(screen_state.get("type_probs"), dict) else {}
        signals = screen_state.get("type_signals") if isinstance(screen_state.get("type_signals"), dict) else {}
        rule = str(signals.get("rule") or signals.get("reason") or "n/a")
        control_kind = str(signals.get("control_kind") or screen_state.get("control_kind") or "unknown")
        options_count = signals.get("options_count")
        kinds = signals.get("controls_kinds")

        mode_hint = "radio(single-choice)"
        if qtype == "multi":
            mode_hint = "checkbox(multi-choice)"
        elif qtype in {"dropdown", "dropdown_scroll"}:
            mode_hint = "dropdown(select)"
        elif qtype == "text":
            mode_hint = "text(input)"
        elif qtype in {"triple", "mixed"}:
            mode_hint = "mixed/compound"

        parts: List[str] = [
            f"detected={qtype}",
            f"block={active_block_type}",
            f"layout={layout_type}",
            f"mode={mode_hint}",
            f"op={op}",
            f"conf={conf:.3f}",
            f"margin={margin:.3f}",
            f"source={source}",
            f"rule={rule}",
            f"control_kind={control_kind}",
        ]
        if isinstance(options_count, int):
            parts.append(f"options={options_count}")
        if isinstance(kinds, list) and kinds:
            parts.append(f"dom_kinds={','.join(str(k) for k in kinds[:6])}")
        if probs:
            top = sorted(((str(k), float(v)) for k, v in probs.items()), key=lambda kv: kv[1], reverse=True)[:3]
            parts.append("top_probs=" + ",".join(f"{k}:{v:.2f}" for k, v in top))
        ev = {
            "q_cnt": signals.get("question_count"),
            "opt_cnt": signals.get("option_count"),
            "has_next": signals.get("has_next"),
            "has_select": signals.get("has_select"),
            "has_input": signals.get("has_input"),
        }
        parts.append("evidence=" + ",".join(f"{k}={ev[k]}" for k in ev if ev[k] is not None))
        try:
            self.logger("[QUIZ_TYPE] " + " | ".join(parts))
        except Exception:
            self._log("QUIZ_TYPE " + " | ".join(parts))
        if str(os.environ.get("FULLBOT_QUIZ_TYPE_DEBUG", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}:
            qsplit = screen_state.get("question_split") if isinstance(screen_state.get("question_split"), dict) else {}
            qtxt = str(qsplit.get("question") or screen_state.get("question_text") or "").strip()
            answers = qsplit.get("answers") if isinstance(qsplit.get("answers"), list) else []
            ans = " ".join(f"[{str(a)}]" for a in answers[:12])
            self.logger(f"[QUESTION_SPLIT] question={qtxt or '<none>'} answers={ans or '[]'}")

    def _log_action_chain(self, *, screen_state: Dict[str, Any], trace: Dict[str, Any], actions: List[Dict[str, Any]]) -> None:
        if not isinstance(trace, dict):
            return
        expectation = trace.get("post_action_expectation") if isinstance(trace.get("post_action_expectation"), dict) else {}
        block = screen_state.get("active_block") if isinstance(screen_state.get("active_block"), dict) else {}
        first_action = actions[0] if actions and isinstance(actions[0], dict) else {}
        parts = [
            f"type={trace.get('chain_type') or 'unknown'}",
            f"stage={trace.get('stage') or 'n/a'}",
            f"block={block.get('block_family') or screen_state.get('active_block_type') or screen_state.get('detected_quiz_type') or 'unknown'}",
            f"answers={int(block.get('answer_count') or len(screen_state.get('options') or []))}",
            f"next={1 if screen_state.get('next_bbox') else 0}",
            f"first={first_action.get('kind') or 'noop'}",
        ]
        reason = str(first_action.get("reason") or "")
        if reason:
            parts.append(f"reason={reason}")
        if expectation:
            parts.append(
                "expect="
                + ",".join(
                    [
                        f"answer={1 if expectation.get('expect_answer_state_change') else 0}",
                        f"next={1 if expectation.get('expect_next_click') else 0}",
                        f"qchange={1 if expectation.get('expect_question_change') else 0}",
                        f"auto={1 if expectation.get('expect_auto_next') else 0}",
                    ]
                )
            )
        try:
            self.logger("[ACTION_CHAIN] " + " | ".join(parts))
        except Exception:
            self._log("ACTION_CHAIN " + " | ".join(parts))

    def _log_verify(self, *, transition: Dict[str, Any], trace: Dict[str, Any]) -> None:
        if not isinstance(transition, dict):
            return
        expectation = trace.get("post_action_expectation") if isinstance(trace.get("post_action_expectation"), dict) else {}
        parts = [
            f"prev={transition.get('previous_action_kind') or 'none'}",
            f"qchange={1 if transition.get('question_changed') else 0}",
            f"qa30={1 if transition.get('qa_changed_30') else 0}",
            f"ratio={float(transition.get('qa_change_ratio') or 0.0):.2f}",
            f"success={1 if transition.get('success') else 0}",
        ]
        if expectation:
            parts.append(f"expected_stage={expectation.get('stage') or 'n/a'}")
        try:
            self.logger("[VERIFY] " + " | ".join(parts))
        except Exception:
            self._log("VERIFY " + " | ".join(parts))

    def _build_quiz_brain_state(
        self,
        *,
        prev_state: Dict[str, Any],
        screen_state: Dict[str, Any],
        question_data: Optional[Dict[str, Any]],
        summary_data: Optional[Dict[str, Any]],
        controls_data: Optional[Dict[str, Any]],
        resolved_answer: Dict[str, Any],
        actions: List[Dict[str, Any]],
        legacy_action: str,
        trace: Dict[str, Any],
        transition: Dict[str, Any],
        fallback_used: bool,
        page_data: Optional[Dict[str, Any]],
        target_bbox: Optional[List[float]],
    ) -> Dict[str, Any]:
        attempts = dict((prev_state or {}).get("attempt_counts") or {})
        answered_signature_memory, answered_signature_flags = _update_answered_signature_memory(
            prev_state=prev_state,
            transition=transition,
            screen_state=screen_state,
        )
        last_action = {}
        if actions:
            first = actions[0]
            expected_values = list(resolved_answer.get("correct_answers") or [])
            post_action_expectation = trace.get("post_action_expectation") if isinstance(trace, dict) else None
            if not isinstance(post_action_expectation, dict):
                post_action_expectation = {}
            target_text = str((expected_values[0] if expected_values else "") or first.get("target_text") or first.get("value") or "").strip()
            action_id = build_action_id(
                screen_signature=str(screen_state.get("screen_signature") or ""),
                question_signature=str(screen_state.get("active_question_signature") or ""),
                first_action_kind=str(first.get("kind") or ""),
                reason=str(first.get("reason") or ""),
                expected_values=expected_values,
            )
            target_instance_id = build_target_instance_id(
                question_signature=str(screen_state.get("active_question_signature") or ""),
                action_kind=str(first.get("kind") or ""),
                target_text=target_text,
                target_bbox=target_bbox or [],
            )
            last_action = {
                "schema_version": SCHEMA_VERSION,
                "action_id": action_id,
                "target_instance_id": target_instance_id,
                "expectation_id": str(post_action_expectation.get("expectation_id") or ""),
                "kind": "next" if legacy_action == "click_next" else ("answer" if legacy_action == "click_answer" else first.get("kind")),
                "raw_kind": first.get("kind"),
                "reason": first.get("reason"),
                "question_signature": screen_state.get("active_question_signature"),
                "page_signature": screen_state.get("page_signature") or (page_data or {}).get("page_signature"),
                "target_text": target_text,
                "target_bbox": target_bbox,
                "expected_values": expected_values,
                "post_action_expectation": post_action_expectation,
                "bbox": target_bbox,
                "timestamp": time.time(),
            }
            attempt_key = "|".join(
                [
                    str(screen_state.get("screen_signature") or ""),
                    str(screen_state.get("active_question_signature") or ""),
                    str(first.get("kind") or ""),
                    ",".join(expected_values),
                    str(first.get("combo") or ""),
                ]
            )
            attempts[attempt_key] = int(attempts.get(attempt_key) or 0) + 1
        return {
            "schema_version": SCHEMA_VERSION,
            "timestamp": time.time(),
            "quiz_mode": True,
            "question_text": str(screen_state.get("question_text") or ""),
            "question_hash": str(screen_state.get("active_question_signature") or ""),
            "detected_quiz_type": str(screen_state.get("detected_quiz_type") or ""),
            "detected_operational_type": str(screen_state.get("detected_operational_type") or ""),
            "type_confidence": float(screen_state.get("type_confidence") or 0.0),
            "type_source": str(screen_state.get("type_source") or ""),
            "question_changed": bool(transition.get("question_changed")),
            "answer_clicked": legacy_action == "click_answer",
            "next_clicked": legacy_action == "click_next",
            "has_answers": bool(screen_state.get("options") or resolved_answer.get("question_type") in {"text", "dropdown", "dropdown_scroll"}),
            "has_next": bool(screen_state.get("next_bbox")),
            "recommended_action": legacy_action,
            "sources": {
                "question_json": str(self.question_path),
                "summary_json": str(summary_data.get("_source")) if isinstance(summary_data, dict) and summary_data.get("_source") else None,
                "controls_json": str(self.controls_path) if self.controls_path else None,
                "page_json": str(self.page_path) if self.page_path else None,
            },
            "objects": {
                "question": {
                    "text": screen_state.get("question_text"),
                    "bbox": (((screen_state.get("active_block") or {}).get("bbox")) if screen_state.get("active_block") else ((screen_state.get("questions") or [{}])[0].get("bbox") if screen_state.get("questions") else None)),
                },
                "answers": screen_state.get("options") or [],
                "next": {"bbox": screen_state.get("next_bbox")} if screen_state.get("next_bbox") else None,
                "cookies": None,
            },
            "actions": actions,
            "screen_state": screen_state,
            "blocks": screen_state.get("blocks") or [],
            "active_block": screen_state.get("active_block") or {},
            "element_roles": screen_state.get("element_roles") or [],
            "resolved_answer": resolved_answer,
            "fallback_used": bool(fallback_used),
            "screen_target_source": "screen_only",
            "screen_parse_quality": str(screen_state.get("screen_parse_quality") or "unknown"),
            "screen_block_reason": str(((actions[0] if actions else {}) or {}).get("reason") or ""),
            "page_signature": screen_state.get("page_signature") or (page_data or {}).get("page_signature"),
            "last_action": last_action,
            "last_screen_signature": screen_state.get("screen_signature"),
            "post_action_expectation": (trace.get("post_action_expectation") if isinstance(trace, dict) else {}),
            "attempt_counts": attempts,
            "answered_signature_memory": answered_signature_memory,
            "answered_signature_loop_detected": bool(answered_signature_flags.get("revisit_detected")),
            "answered_signature_revisit_count": int(answered_signature_flags.get("revisit_count") or 0),
            "transition": transition,
            "controls_meta": (controls_data or {}).get("meta") if isinstance(controls_data, dict) else {},
        }

    def _write_quiz_debug(self, *, screen_state: Dict[str, Any], trace: Dict[str, Any], resolved: Dict[str, Any]) -> None:
        if self.current_run_dir is None:
            return
        try:
            self.current_run_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        parse_path = self.current_run_dir / "screen_quiz_parse.json"
        try:
            parse_payload = {
                "schema_version": SCHEMA_VERSION,
                "screen_state": screen_state,
                "resolved_answer": resolved,
                "trace": trace,
            }
            parse_path.write_text(json.dumps(parse_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        # v2 compact artifact for runtime classifier observability.
        parse_v2_path = self.current_run_dir / "screen_quiz_parse_v2.json"
        try:
            parse_v2 = {
                "schema_version": SCHEMA_VERSION,
                "global_type": screen_state.get("detected_quiz_type"),
                "active_block_type": screen_state.get("active_block_type"),
                "layout_type": screen_state.get("layout_type"),
                "operational_type": screen_state.get("detected_operational_type"),
                "confidence": screen_state.get("type_confidence"),
                "margin": screen_state.get("decision_margin"),
                "type_probs": screen_state.get("type_probs") or {},
                "evidence": screen_state.get("type_signals") or {},
                "blocks": screen_state.get("blocks") or [],
                "active_block": screen_state.get("active_block") or {},
                "element_roles": screen_state.get("element_roles") or [],
                "active_question_id": screen_state.get("active_question_id"),
                "active_question_signature": screen_state.get("active_question_signature"),
                "parse_signature_v2": screen_state.get("parse_signature_v2"),
                "question_split": screen_state.get("question_split") or {},
                "screen_signature": screen_state.get("screen_signature"),
                "post_action_expectation": trace.get("post_action_expectation") if isinstance(trace, dict) else {},
            }
            parse_v2_path.write_text(json.dumps(parse_v2, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        features_path = self.current_run_dir / "quiz_type_features.json"
        try:
            features_payload = {
                "schema_version": SCHEMA_VERSION,
                "ts": time.time(),
                "features": screen_state.get("quiz_type_features") or {},
                "global_type": screen_state.get("detected_quiz_type"),
                "confidence": screen_state.get("type_confidence"),
                "margin": screen_state.get("decision_margin"),
                "parse_signature_v2": screen_state.get("parse_signature_v2"),
                "active_block": screen_state.get("active_block") or {},
                "block_count": len(screen_state.get("blocks") or []),
            }
            features_path.write_text(json.dumps(features_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        roles_path = self.current_run_dir / "element_roles.json"
        try:
            roles_payload = {
                "schema_version": SCHEMA_VERSION,
                "ts": time.time(),
                "screen_signature": screen_state.get("screen_signature"),
                "roles": screen_state.get("element_roles") or [],
            }
            roles_path.write_text(json.dumps(roles_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        blocks_path = self.current_run_dir / "screen_blocks.json"
        try:
            blocks_payload = {
                "schema_version": SCHEMA_VERSION,
                "ts": time.time(),
                "screen_signature": screen_state.get("screen_signature"),
                "global_type": screen_state.get("detected_quiz_type"),
                "active_block_id": screen_state.get("active_block_id"),
                "active_block": screen_state.get("active_block") or {},
                "blocks": screen_state.get("blocks") or [],
            }
            blocks_path.write_text(json.dumps(blocks_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        action_trace_path = self.current_run_dir / "action_trace.json"
        try:
            action_payload = {
                "schema_version": SCHEMA_VERSION,
                "ts": time.time(),
                "screen_signature": screen_state.get("screen_signature"),
                "question_text": screen_state.get("question_text"),
                "trace": trace,
                "resolved_answer": resolved,
            }
            action_trace_path.write_text(json.dumps(action_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        trace_path = self.current_run_dir / "quiz_trace.jsonl"
        try:
            line = {
                "schema_version": SCHEMA_VERSION,
                "ts": time.time(),
                "question_text": screen_state.get("question_text"),
                "screen_signature": screen_state.get("screen_signature"),
                "trace": trace,
                "resolved_answer": resolved,
            }
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _quiz_mode_enabled(self) -> bool:
        return str(os.environ.get("FULLBOT_QUIZ_MODE", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}

    def _quiz_answer_cache(self) -> Path:
        env_path = str(os.environ.get("FULLBOT_QUIZ_ANSWER_CACHE", "") or "").strip()
        if env_path:
            return Path(env_path)
        return self.quiz_answer_cache

    def _screen_site_consistency(self) -> Optional[float]:
        question_data = load_json(self.question_path, self._log)
        page_data = load_json(self.page_path, self._log) if self.page_path else None
        if not isinstance(question_data, dict) or not isinstance(page_data, dict):
            return None

        question_host = _norm_host(question_data.get("url"))
        page_host = _norm_host(page_data.get("url"))
        question_title = _norm_text(question_data.get("title"))
        page_title = _norm_text(page_data.get("title"))
        question_viewport = question_data.get("viewport") if isinstance(question_data.get("viewport"), dict) else {}
        page_viewport = page_data.get("viewport") if isinstance(page_data.get("viewport"), dict) else {}

        parts: List[float] = []
        if question_host and page_host:
            host_match = question_host == page_host or question_host.endswith(page_host) or page_host.endswith(question_host)
            parts.append(0.6 if host_match else 0.0)
        if question_title and page_title:
            title_match = question_title == page_title or question_title in page_title or page_title in question_title
            parts.append(0.25 if title_match else 0.0)
        if question_viewport and page_viewport:
            viewport_match = (
                int(question_viewport.get("width") or 0) == int(page_viewport.get("width") or 0)
                and int(question_viewport.get("height") or 0) == int(page_viewport.get("height") or 0)
            )
            parts.append(0.15 if viewport_match else 0.0)
        if not parts:
            return None
        return round(max(0.0, min(1.0, sum(parts))), 4)

    def _log(self, message: str) -> None:
        try:
            self.logger(f"[brain] {message}")
        except Exception:
            pass

    def _error(self, message: str) -> None:
        try:
            self.logger(f"[ERROR] {message}")
        except Exception:
            pass
