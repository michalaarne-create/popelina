from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
import traceback


def _write_loop_trace(current_run_dir: Path, row: dict[str, Any]) -> None:
    current_run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = current_run_dir / "loop_trace.jsonl"
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_loop_summary(current_run_dir: Path, *, iterations: int, successes: int, failures: int, last_result: dict[str, Any]) -> None:
    current_run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "iterations": int(iterations),
        "successes": int(successes),
        "failures": int(failures),
        "last_result": last_result,
    }
    (current_run_dir / "loop_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run_loop(
    *,
    args: Any,
    trigger_event: Any,
    command_queue: Any,
    state: dict,
    pipeline_iteration,
    drain_manual_commands,
    ensure_control_agent,
    cancel_hover_fallback_timer,
    wait_for_p_in_console,
    log,
    debug,
    update_overlay_status,
) -> None:
    loop_idx = 0
    successes = 0
    failures = 0
    current_run_dir = Path(state["current_run_dir"]) if state.get("current_run_dir") else None
    while True:
        if not args.auto:
            if state.get("console_p_mode"):
                log(f"[INFO] Waiting for 'P' in console to start iteration #{loop_idx + 1}...")
                update_overlay_status(f"Waiting for 'P' in console (iteration {loop_idx + 1})")
                wait_for_p_in_console()
            else:
                log(f"[INFO] Waiting for hotkey 'P' to start iteration #{loop_idx + 1}...")
                update_overlay_status(f"Waiting for 'P' (iteration {loop_idx + 1})")
                while not trigger_event.wait(timeout=0.2):
                    state["recorder_proc"] = drain_manual_commands(command_queue, args, state["recorder_proc"])
                trigger_event.clear()
        else:
            state["recorder_proc"] = drain_manual_commands(command_queue, args, state["recorder_proc"])

        loop_idx += 1
        if state.get("control_agent_proc") is None and not args.safe_test:
            state["control_agent_proc"] = ensure_control_agent()

        cancel_hover_fallback_timer()
        log(f"[INFO] Iteration {loop_idx} start")
        update_overlay_status(f"Iteration {loop_idx} started")
        iter_start = time.perf_counter()
        try:
            if state.get("debug_mode"):
                debug(f"pipeline_iteration start idx={loop_idx}")
            result = pipeline_iteration(
                loop_idx,
                input_image=Path(args.input_image) if args.input_image else None,
                fast_skip=args.fast_skip,
            )
            if state.get("debug_mode"):
                debug(f"pipeline_iteration end idx={loop_idx}")
            if result is not None:
                last_result = {
                    "loop_idx": int(getattr(result, "loop_idx", loop_idx)),
                    "status": str((getattr(result, "metadata", {}) or {}).get("status") or ""),
                    "rating_ok": bool(getattr(result, "rating_ok", False)),
                    "decision_action": getattr(result, "decision_action", None),
                    "elapsed_s": float(getattr(result, "elapsed_s", 0.0) or 0.0),
                    "summary_path": str(getattr(result, "summary_path", "") or ""),
                    "region_json_path": str(getattr(result, "region_json_path", "") or ""),
                    "screenshot_path": str(getattr(result, "screenshot_path", "") or ""),
                }
                status = last_result["status"]
                if status == "completed":
                    successes += 1
                else:
                    failures += 1
                if current_run_dir is not None:
                    _write_loop_trace(current_run_dir, last_result)
                    _write_loop_summary(
                        current_run_dir,
                        iterations=loop_idx,
                        successes=successes,
                        failures=failures,
                        last_result=last_result,
                    )
        except Exception as exc:
            log(f"[ERROR] Pipeline iteration failed: {exc}")
            if state.get("debug_mode"):
                debug(traceback.format_exc())
            failures += 1
            if current_run_dir is not None:
                error_result = {
                    "loop_idx": int(loop_idx),
                    "status": "exception",
                    "rating_ok": False,
                    "decision_action": None,
                    "elapsed_s": 0.0,
                    "error": str(exc),
                }
                _write_loop_trace(current_run_dir, error_result)
                _write_loop_summary(
                    current_run_dir,
                    iterations=loop_idx,
                    successes=successes,
                    failures=failures,
                    last_result=error_result,
                )
        state["recorder_proc"] = drain_manual_commands(command_queue, args, state["recorder_proc"])

        if args.loop_count and loop_idx >= args.loop_count:
            break
        if args.auto:
            delay = max(0.0, args.interval - (time.perf_counter() - iter_start))
            if delay:
                time.sleep(delay)
        else:
            state["recorder_proc"] = drain_manual_commands(command_queue, args, state["recorder_proc"])
