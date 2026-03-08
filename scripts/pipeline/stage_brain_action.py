from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Optional


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
    log: Callable[[str], None],
    update_overlay_status: Callable[[str], None],
) -> None:
    t0 = time.perf_counter()
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
        screenshot = find_screenshot_for_summary(summary_path) or screenshot_path
        send_best_click(summary_path, screenshot)
    elif decision.recommended_action in ("click_answer", "fallback_random"):
        if not bool(getattr(decision, "brain_state", {}).get("quiz_mode")):
            send_random_click(summary_path, screenshot_path)
        else:
            log("[INFO] Quiz mode suppressed random click fallback.")
    else:
        log("[INFO] Brain recommended idle; no click sent.")
    update_overlay_status("rating completed.")
    log(f"[TIMER] stage_brain_action {time.perf_counter() - t0:.3f}s action={decision.recommended_action}")
