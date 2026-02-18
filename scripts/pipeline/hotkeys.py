from __future__ import annotations

import os
from typing import Any, Optional


def start_hotkey_listener(
    event: Any,
    command_queue: Any,
    *,
    log,
    debug,
    update_overlay_status,
    globals_fn,
    is_debug_mode,
) -> Optional[Any]:
    try:
        from pynput import keyboard  # type: ignore
    except Exception as exc:
        log(f"[WARN] Hotkey listener unavailable (pynput import failed: {exc}). Falling back to auto mode.")
        return None

    def on_press(key):
        try:
            if not key.char:
                return
            ch = key.char.lower()
            if ch == "p":
                globals_fn()["_hover_fallback_allowed"] = True
                log("[INFO] Hotkey 'P' pressed - starting pipeline.")
                if bool(is_debug_mode()):
                    debug("Hotkey P captured by listener, setting trigger_event.")
                event.set()
                update_overlay_status("Hotkey 'P' pressed - pipeline starting.")
            elif ch == "o":
                log("[INFO] Hotkey 'O' pressed - manual region_grow.")
                update_overlay_status("Hotkey 'O' pressed - region_grow queued.")
                command_queue.put("region")
            elif ch == "l":
                log("[INFO] Hotkey 'L' pressed - recorder launch requested.")
                update_overlay_status("Hotkey 'L' pressed - recorder queued.")
                command_queue.put("recorder")
            elif ch == "i":
                log("[INFO] Hotkey 'I' pressed - manual rating.")
                update_overlay_status("Hotkey 'I' pressed - rating queued.")
                command_queue.put("rating")
            elif ch == "k":
                log("[INFO] Hotkey 'K' pressed - control agent best click.")
                update_overlay_status("Hotkey 'K' pressed - control agent queued.")
                command_queue.put("control")
            elif ch == '"':
                if bool(globals_fn().get("_iteration_in_progress")):
                    globals_fn()["_abort_iteration_requested"] = True
                    command_queue.put("abort")
                    log("[WARN] Hotkey '\"' pressed - aborting current iteration.")
                    update_overlay_status("Abort requested for current iteration.")
                else:
                    log("[INFO] Hotkey '\"' ignored (no active iteration).")
            elif ch == "}":
                log("[WARN] Hard exit requested via '}' hotkey.")
                os._exit(0)
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    log("[INFO] Hotkeys: P=pipeline, O=region_grow, L=recorder, I=rating, K=control agent, \"=abort iteration.")
    update_overlay_status("Hotkeys ready (P/O/L/I/K/\").")
    return listener


def wait_for_p_in_console() -> None:
    if os.name == "nt":
        try:
            import msvcrt  # type: ignore

            while True:
                ch = msvcrt.getwch()
                if isinstance(ch, str) and ch.lower() == "p":
                    return
        except Exception:
            pass
    while True:
        line = input("Press P + Enter to run one iteration: ").strip().lower()
        if line == "p":
            return
