from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
import traceback


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
    msvcrt_mod = None
    if os.name == "nt":
        try:
            import msvcrt as _msvcrt  # type: ignore

            msvcrt_mod = _msvcrt
        except Exception:
            msvcrt_mod = None

    def _poll_console_p_nonblocking() -> bool:
        if msvcrt_mod is None:
            return False
        try:
            if not msvcrt_mod.kbhit():
                return False
            ch = msvcrt_mod.getwch()
            return isinstance(ch, str) and ch.lower() == "p"
        except Exception:
            return False

    loop_idx = 0
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
                    if _poll_console_p_nonblocking():
                        log("[INFO] Console key 'P' detected — starting pipeline.")
                        update_overlay_status("Console key 'P' detected — pipeline starting.")
                        trigger_event.set()
                        break
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
            pipeline_iteration(
                loop_idx,
                input_image=Path(args.input_image) if args.input_image else None,
                fast_skip=args.fast_skip,
            )
            if state.get("debug_mode"):
                debug(f"pipeline_iteration end idx={loop_idx}")
        except Exception as exc:
            log(f"[ERROR] Pipeline iteration failed: {exc}")
            if state.get("debug_mode"):
                debug(traceback.format_exc())
        state["recorder_proc"] = drain_manual_commands(command_queue, args, state["recorder_proc"])

        if args.loop_count and loop_idx >= args.loop_count:
            break
        if args.auto:
            delay = max(0.0, args.interval - (time.perf_counter() - iter_start))
            if delay:
                time.sleep(delay)
        else:
            state["recorder_proc"] = drain_manual_commands(command_queue, args, state["recorder_proc"])
