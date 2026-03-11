from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class BrainDispatchResult:
    summary_path: Path
    decision: Any
    is_new_question: bool


def _clip(text: str, max_len: int = 120) -> str:
    s = str(text or "").strip().replace("\n", " ")
    return s if len(s) <= max_len else (s[: max(0, max_len - 1)] + "…")


def collect_and_dispatch_to_brain(
    *,
    screenshot_path: Path,
    json_path: Path,
    rate_results_dir: Path,
    rate_results_debug_dir: Path,
    rate_results_current_dir: Path,
    rate_results_debug_current_dir: Path,
    rate_summary_dir: Path,
    rate_summary_current_dir: Path,
    write_current_artifact: Callable[[Path, Path, Optional[str]], Optional[Path]],
    brain_agent: Any,
    log: Callable[[str], None],
) -> Optional[BrainDispatchResult]:
    """
    Collect rating outputs, resolve summary path and dispatch it to brain agent.
    """
    try:
        stem = screenshot_path.stem
        rated_path = rate_results_dir / f"{stem}_rated.json"
        rated_debug_path = rate_results_debug_dir / f"{stem}_rated_debug.json"
        if rated_path.exists():
            write_current_artifact(rated_path, rate_results_current_dir, rated_path.name)
        if rated_debug_path.exists():
            write_current_artifact(rated_debug_path, rate_results_debug_current_dir, rated_debug_path.name)
    except Exception:
        pass

    summary_candidates = [
        rate_summary_dir / f"{screenshot_path.stem}_summary.json",
        rate_summary_dir / f"{json_path.stem}_summary.json",
    ]
    if json_path.stem.endswith("_rg_small"):
        alt_stem = json_path.stem[: -len("_rg_small")]
        summary_candidates.append(rate_summary_dir / f"{alt_stem}_summary.json")

    summary_path: Optional[Path] = None
    for cand in summary_candidates:
        if cand.exists():
            summary_path = cand
            break

    turbo_mode = str(os.environ.get("FULLBOT_TURBO_MODE", "1") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if summary_path is None and turbo_mode:
        # In turbo mode prefer deterministic fast summary over "latest *_summary.json".
        try:
            fast_candidates = [
                json_path.with_name(f"{json_path.stem}_fast_summary.json"),
                json_path.parent.parent / "region_grow_current" / "fast_summary.json",
            ]
            for cand in fast_candidates:
                if cand.exists():
                    summary_path = cand
                    log(f"[INFO] Using fast summary fallback: {cand.name}")
                    break
        except Exception:
            summary_path = None

    if summary_path is None and (not turbo_mode):
        # Fallback: rating can emit summary keyed by `image` basename (often `screenshot.png`)
        # rather than current screenshot stem (`screen_YYYY...`).
        try:
            recent = sorted(
                (p for p in rate_summary_dir.glob("*_summary.json") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if recent:
                summary_path = recent[0]
                log(
                    f"[WARN] Summary name mismatch; using latest summary: {summary_path.name}"
                )
        except Exception:
            summary_path = None

    if summary_path is None:
        log(f"[WARN] Summary missing at {summary_candidates[0]}; skipping brain decision.")
        return None

    try:
        write_current_artifact(summary_path, rate_summary_current_dir, summary_path.name)
    except Exception:
        pass

    decision = brain_agent.decide(
        summary_path,
        region_json_path=json_path,
        screenshot_path=screenshot_path,
    )
    try:
        state = (getattr(decision, "screen_state", None) or {}) if decision is not None else {}
        resolved = {}
        trace = (getattr(decision, "trace", None) or {}) if decision is not None else {}
        if isinstance(trace, dict):
            resolved = trace.get("resolved") or {}
        control_kind = str(state.get("control_kind") or ((resolved or {}).get("question_type") or "unknown"))
        detected_type = str(state.get("detected_quiz_type") or control_kind)
        detected_op = str(state.get("detected_operational_type") or "")
        type_conf = float(state.get("type_confidence") or 0.0)
        type_source = str(state.get("type_source") or "screen")
        q_text = _clip(str(state.get("question_text") or ""))
        q_sig = str(state.get("active_question_signature") or "")
        has_next = int(bool(state.get("next_bbox")))
        options_n = len(state.get("options") or []) if isinstance(state.get("options"), list) else 0
        source = str(getattr(decision, "answer_source", "") or (resolved or {}).get("source") or "-")
        log(
            "[INFO] Brain quiz parse: "
            f"type={detected_type} op={detected_op or control_kind} conf={type_conf:.2f} src={type_source} "
            f"has_next={has_next} options={options_n} "
            f"action={getattr(decision, 'recommended_action', 'idle')} source={source} "
            f"qsig={q_sig[:12] if q_sig else '-'} text='{q_text}'"
        )
    except Exception:
        pass
    is_new_question = bool((getattr(decision, "brain_state", {}) or {}).get("question_changed"))
    return BrainDispatchResult(
        summary_path=summary_path,
        decision=decision,
        is_new_question=is_new_question,
    )
